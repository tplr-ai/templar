# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# fmt: off

# Global imports.
import sys
import time
import wandb
import torch
import random
import asyncio
import argparse
import threading
import bittensor as bt
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import local package.
import tplr

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Globals: determined by master.
SPEC_VERSION = 4 # Run version.
PROJECT = 'templar' # wandb project.
SEQUENCE_LENGTH = 1024 # global sequence length.
PAGES_PER_WINDOW = 2 # Pages to train on (and be evaluated on each window.)
BATCH_SIZE = 8 # global batch size.
LEARNING_RATE = 0.001 # global learning rate.
BLOCKS_PER_WINDOW = 2 # blocks per window step.
WINDOWS_PER_SYNC = 100 # Step Windows before sync state occurs.
MOMENTUM_DECAY = 0.999 # momentum deacy rate.
TOPK_COMPRESSION = 32 # DeMo Topk Compression.
TARGET_CHUNK = 64 # DeMo chunk size.
SCORES_ALPHA = 0.001 # Scores moving average.
WINDOWS_PER_WEIGHTS = 10 # Windows before validator sets weights on chain.

class Miner:
    
    # Command line config items.
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Miner script')
        parser.add_argument('--netuid', type=int, default=229, help='Bittensor network UID.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
        parser.add_argument('--is_validator', action='store_true', help='If validator, turn on to run evals rather than train for incentive.')
        parser.add_argument('--random', action='store_true', help='Trains on a random page instead of correctly assigned.')
        parser.add_argument('--peers', type=int, nargs='+', default=[], help='List of UIDs to peer with. e.g., --uids 1 2 3')
        bt.wallet.add_args( parser )
        bt.subtensor.add_args( parser )
        bt.logging.add_args( parser )
        config = bt.config( parser )
        if config.debug: tplr.debug()
        if config.trace: tplr.trace()
        return config
    
    def __init__(self):
        # Init config from command line.
        self.config = Miner.config()
        
        # Init bittensor objects.
        self.wallet = bt.wallet( config = self.config )
        self.subtensor = bt.subtensor( config = self.config )
        self.metagraph = self.subtensor.metagraph( self.config.netuid )
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            tplr.logger.error(f'\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}[/bold]. You need to register first with: [blue]`btcli subnet register`[/blue]\n')
            sys.exit()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        tplr.logger.info('\n' + '-' * 40 + ' Objects ' + '-' * 40)
        tplr.logger.info(f'\n{self.wallet}\n{self.subtensor}\n{self.metagraph}\nuid: {self.uid}')
        
        # Init peers.
        self.peers = list( self.metagraph.uids) if self.config.peers == [] else self.config.peers
        if self.uid not in self.peers: self.peers.append( self.uid ) # Add my self to peers.
        tplr.logger.info(f'peers: {self.peers}')

        # Initialize the model with random weights.
        self.model = GPT2LMHeadModel(GPT2LMHeadModel.config_class())
        self.model.to(self.config.device)
        
        # Init tokenizer.
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Init optimizer.
        self.momentum = {}
        self.optimizer = optim.SGD(self.model.parameters(), lr = LEARNING_RATE)          
        for n, p in self.model.named_parameters():
            self.momentum[n] = torch.zeros_like(p)
        
        # Init compression.
        self.transformer = tplr.compress.TransformDCT( self.model, target_chunk = TARGET_CHUNK )
        self.compressor = tplr.compress.CompressDCT()
        
        # Init state params.
        self.stop_event = asyncio.Event()
        self.current_block = self.subtensor.block
        self.current_window = int( self.current_block / BLOCKS_PER_WINDOW )
        
        # Init scores.
        self.scores = torch.zeros(self.metagraph.n, dtype=torch.float32)
        
        # Init wandb.
        if self.config.use_wandb:
            # Delete all runs with my name and create a new one.
            try:
                for run in wandb.Api().runs(path=PROJECT):
                    if run.name == f'M{self.uid}': run.delete()
            except: pass
            wandb.init(project=PROJECT, resume='allow', name=f'M{self.uid}', config=self.config)
        
    # Main training loop.
    async def run( self ):

        # Start background block listener.       
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(target=self.block_listener, args=(self.loop,), daemon=True).start()

        # Run until stopped.
        while True:
            
            # Record the window we are on.
            step_window = self.current_window
            # Get the uid to seed data (if validator, take random from peers.)
            step_uid = self.uid if not self.config.is_validator else random.choice(self.config.peers)
            tplr.logger.info('\n' + '-' * 40 + f' Window: {step_window} ' + '-' * 40)
            
            # Optionally sync state.
            if step_window % WINDOWS_PER_SYNC == 0:
                tplr.logger.info(f"Sync globally")
                gather_result = await tplr.comms.gather(
                    state_dict = self.model.state_dict(),
                    my_uid = self.uid,
                    uids = self.peers,
                    window = int(self.current_window/WINDOWS_PER_SYNC),
                    key = 'model',
                    timeout = 30,
                    device = self.config.device
                )
                # Take median of all peers state.
                state_dict = {name: torch.median(torch.stack(gather_result[name]), dim=0)[0] for name in gather_result}
                # Load state into model.
                self.model.load_state_dict(state_dict)
                tplr.logger.info(f"Done global sync.")

            # Get the pages for this window.
            pages = await tplr.dataset.DatasetLoader.next_pages(
                offset = step_window,
                n_pages = PAGES_PER_WINDOW,
                seed = self.metagraph.hotkeys[ step_uid ] if not self.config.random else random.randint(10000) # Select seed from step_uid.
            )            
            loader = await tplr.dataset.DatasetLoader.create(
                batch_size = BATCH_SIZE,
                sequence_length = SEQUENCE_LENGTH,
                pages_info = pages,
                tokenizer = self.tokenizer
            )   
            tplr.logger.info(f"Pages: {[p[1] for p in pages]} for UID: {step_uid} and Window: {step_window}")
            
            # Accumulate gradient.
            tplr.logger.info(f"Start accumulating...")
            self.optimizer.zero_grad()
            self.model.zero_grad()
            for i, batch in enumerate( loader ):
                input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                outputs = self.model(input_ids=input_ids, labels=labels)
                outputs.loss.backward()
                print ('loss:', outputs.loss.item())
                if self.current_window != step_window:
                    break
            tplr.logger.info(f"Stopped accumulating: {i+1} batches with {(i+1) * BATCH_SIZE * SEQUENCE_LENGTH} tokens ")
            # Log to wandb.
            if self.config.use_wandb: wandb.log({f"loss": outputs.loss.item()})
                
            # Reduce gradient using DeMo.
            gradient = {}
            xshapes = {}
            totalks = {}
            transmitted = {}
            for n, p in self.model.named_parameters():
                # Momentum decay
                self.momentum[n].mul_( MOMENTUM_DECAY )
                # Add the grad to the momentum.
                self.momentum[n].add_( p.grad, alpha = LEARNING_RATE )
                # Compress gradient.
                idxs, vals, xshape, totalk = self.compressor.compress(
                    self.transformer.encode(self.momentum[n]), TOPK_COMPRESSION
                )
                # Estimate transmitted gradient.
                transmit_grad = self.transformer.decode(
                    self.compressor.decompress(p, idxs, vals, xshape, totalk)
                )
                # Remove the transmitted from delta (double couting)
                self.momentum[n].sub_(transmit_grad)
                # Add to share_state
                transmitted[ n ] = transmit_grad
                gradient[ n + 'idxs'] = idxs 
                gradient[ n + 'vals'] = vals
                xshapes[ n ] = xshape; totalks[ n ] = totalk

            # All-gather share state from all peers with timeout.
            tplr.logger.info(f"Start gather: {self.peers}")
            gather_result = await tplr.comms.gather(
                state_dict = gradient,
                my_uid = self.uid,
                uids = self.peers,
                window = step_window,
                key = 'gradient',
                timeout = 5,
                device = self.config.device
            )
            
            # Decompress state and apply to grad.
            for n, p in self.model.named_parameters():
                # Decode grad from all nodes
                if self.config.is_validator:
                    # Get gradient for step uid we are evaluating.
                    eval_idx = gather_result[n + 'idxs'][ self.peers.index(step_uid) ]
                    eval_val = gather_result[n + 'vals'][ self.peers.index(step_uid) ]
                    # Decompress their gradinet.
                    their_grad = self.transformer.decode(
                        self.compressor.decompress(p, eval_idx, eval_val, xshapes[ n ], totalks[ n ])
                    )
                    # Get my recreated gradient.
                    my_grad = transmitted[ n ]
                    # Compute cosine sim score.
                    score = torch.nn.functional.cosine_similarity(their_grad.flatten(), my_grad.flatten(), dim=0)
                    # Compute moving scores and weights.
                    self.scores[step_uid] = SCORES_ALPHA * score + (1 - SCORES_ALPHA) * self.scores[step_uid].expand_as(score)
                    self.weights = torch.softmax(self.scores, dim=0)
                    # Log scores and weights to wandb.
                    if self.config.use_wandb:
                        for uid in self.peers:
                            wandb.log({f"s{uid}": self.scores[uid], f"w{uid}": self.weights[uid] })
                    
                # Decompress all gradients in batch form to produce shared gradient.
                new_grad = self.transformer.decode(
                    self.compressor.batch_decompress(
                        p, gather_result[n + 'idxs'], gather_result[n + 'vals'], xshapes[ n ], totalks[ n ]
                    )
                )
                # Set recomputed gathered gradient.
                if p.grad is None: p.grad = new_grad
                else: p.grad.copy_(new_grad)
                # Sign-SGD
                p.grad.sign_()
                    
            # Apply the optimizer step
            tplr.logger.info(f"Finish and step.")
            self.optimizer.step()
            
            # Set weights on the chain based on current weights.
            if self.config.is_validator and step_window % WINDOWS_PER_WEIGHTS == 0:
                
                # Set weights on chain.
                self.subtensor.set_weights(
                    wallet = self.wallet,
                    netuid = self.config.netuid,
                    uids = self.metagraph.uids,
                    weights = self.weights,
                    wait_for_inclusion = False, # Dont wait, fire and forget.
                    wait_for_finalization = False,
                )
                
            # Check for autoupdate every 360 blocks.
            if self.current_block % 360 == 0:
                tplr.optionally_auto_update( SPEC_VERSION )
            
            # Wait for end of window (if not already done.)
            while self.current_window == step_window: time.sleep(0.1)
            
    # Listens for new blocks and sets self.current_block and self.current_window
    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event['header']['number'])
            if int( self.current_block / BLOCKS_PER_WINDOW ) != self.current_window:
                self.current_window = int( self.current_block / BLOCKS_PER_WINDOW ) 
        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler); break
            except Exception as e:
                time.sleep(1) 

# Start miner/validator.
if __name__ == "__main__":
    asyncio.run( Miner().run() )