# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# fmt: off

# Global imports
import os
import sys
import time
import torch
import random
import asyncio
import argparse
import threading
import bittensor as bt
import torch.optim as optim
from transformers import LlamaForCausalLM
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Local imports
import tplr
import tplr.checkpoint

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Neuron:
    
    # Command line config items.
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Miner / Validator script')
        parser.add_argument('--netuid', type=int, default=268, help='Bittensor network UID.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
        parser.add_argument('--is_validator', action='store_true', help='If validator, turn on to run evals rather than train for incentive.')
        parser.add_argument('--random', action='store_true', help='Trains on a random page instead of correctly assigned.')
        parser.add_argument('--peers', type=int, nargs='+', default=[], help='List of UIDs to peer with. e.g., --uids 1 2 3')
        parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to save/load the checkpoint. If None, the path is set to checkpoint-M<UID>.pth.')
        parser.add_argument('--save-location', type=str, default=None, help='Directory to save/load slice files')
        bt.wallet.add_args( parser )
        bt.subtensor.add_args( parser )
        bt.logging.add_args( parser )
        config = bt.config( parser )
        if config.debug:
            tplr.debug()      
        if config.trace:
            tplr.trace()
        return config
    
    def __init__(self):
        tplr.logger.debug("Starting initialization...")

        # Init config from command line
        self.config = Neuron.config()

        # # Init AutoUpdate
        # self.autoupdate = tplr.autoupdate.AutoUpdate()

        # Load hyperparameters
        self.hparams = tplr.load_hparams()
                
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
        tplr.logger.debug("Initialized bittensor objects...")
        tplr.logger.debug("Initializing buckets...")
        # Buckets must
        self.buckets = {}  # Initialize empty dict first


        # Initialize the model with config from hparams
        self.model = LlamaForCausalLM(self.hparams.model_config)
        self.model.to(self.config.device)
        # Print model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tplr.logger.info(f"Total parameters: {total_params:,}")
        tplr.logger.debug("Initialized model...")
        
        # Init tokenizer.
        self.tokenizer = self.hparams.tokenizer
        
        # Init optimizer.
        self.momentum = {}
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.hparams.learning_rate)          
        for n, p in self.model.named_parameters():
            self.momentum[n] = torch.zeros_like(p)
        self.scheduler = tplr.CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.hparams.warmup_steps,
            alpha_f=self.hparams.alpha_f,
            t_max=self.hparams.t_max
        )

        # Init compression.
        self.transformer = tplr.compress.TransformDCT( self.model, target_chunk = self.hparams.target_chunk )
        self.compressor = tplr.compress.CompressDCT()

        # # Set checkpoint path => root dir as argumnet and pass root dir, in the init 
        # if self.config.checkpoint_path is None:
        #     # Default path if none provided
        #     self.checkpoint_path = f"checkpoints/checkpoint-{self.uid}.pth"
        # else:
        #     self.checkpoint_path = self.config.checkpoint_path
            
        # # Create checkpoint directory if it doesn't exist
        # os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        

        # # Initialize checkpoint manager
        # self.checkpoint_manager = tplr.checkpoint.CheckpointManager(
        #     model=self.model,
        #     checkpoint_path=self.checkpoint_path,
        #     wallet=self.wallet,
        #     device=self.config.device,
        #     optimizer=self.optimizer,
        #     scheduler=self.scheduler
        # )
        
        # # Load initial checkpoint
        # tplr.logger.debug("Loading checkpoint...")
        # self.global_step = asyncio.run(
        #     self.checkpoint_manager.load_from_highest_stake(
        #         metagraph=self.metagraph,
        #         buckets=self.buckets,
        #         optimizer=self.optimizer,
        #         scheduler=self.scheduler,
        #         is_validator=False, 
        #         hparams=self.hparams
        #     )
        # )


        # Initialize Comms
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location='/tmp',
            key_prefix='model',
            config=self.config,
            netuid=self.config.netuid,
            metagraph=self.metagraph,
            hparams=self.hparams,
        )

        # Initialize peers with buckets
        if not self.config.peers:
            # Use peers with buckets from ChainManager
            self.peers = self.comms.peers
            tplr.logger.info(f'Filtered peers with buckets: {self.peers}')
        else:
            self.peers = self.config.peers  # Use specified peers

        # Ensure we have at least one peer
        if self.config.is_validator and not self.peers:
            tplr.logger.error(
                "No peers available for validation. Ensure there are miners with buckets registered."
            )
            sys.exit(1)

        # Add self to peers if not already included
        if self.uid not in self.peers:
            self.peers.append(self.uid)

        tplr.logger.info(f'Active peers: {self.peers}')

        # Init state params.
        self.stop_event = asyncio.Event()
        self.current_block = self.subtensor.block
        self.current_window = int( self.current_block / self.hparams.blocks_per_window )
        
        # Init scores.
        self.scores = torch.zeros(self.metagraph.n, dtype=torch.float32)
        
        # Init wandb.
        if self.config.use_wandb:
            self.wandb = tplr.wandb.WandbManager(
                uid=self.uid,
                config=self.config,
                is_validator=self.config.is_validator
            ).run
        
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
            step_uid = self.uid if not self.config.is_validator else random.choice(self.peers)
            tplr.logger.info('\n' + '-' * 40 + f' Window: {step_window} ' + '-' * 40)

            # Checkpoint: every X windows , the validators with the highest stake will comms.put into s3, if model is None
            # wait until until next window that % 100 == 0, just gather validator with max stake
            
            # Optionally sync state. Take this out 
            if step_window % self.hparams.windows_per_sync == 0:
                tplr.logger.info("Sync globally")
                # This gather op is way too slow
                # When a miner joins the the network , wait until a new checkpoint has being put up by the validator
                gather_result = await self.comms.gather(
                    state_dict = self.model.state_dict(),
                    my_uid = self.uid,
                    uids = self.peers,
                    window = int(self.current_window/self.hparams.windows_per_sync),
                    key = 'model',
                    timeout = 30,
                    device = self.config.device
                )
                # Take mean of all peers state
                state_dict = {name: torch.mean(torch.stack(gather_result[name]), dim=0) for name in gather_result}
                # Load state into model.
                self.model.load_state_dict(state_dict)
                tplr.logger.info("Done global sync.")

            # Get the pages for this window.
            pages = await tplr.dataset.DatasetLoader.next_pages(
                offset = step_window,
                n_pages = self.hparams.pages_per_window,
                seed = self.metagraph.hotkeys[ step_uid ] if not self.config.random else random.randint(10000) # Select seed from step_uid.
            )            
            loader = await tplr.dataset.DatasetLoader.create(
                batch_size = self.hparams.batch_size,
                sequence_length = self.hparams.sequence_length,
                pages_info = pages,
                tokenizer = self.tokenizer
            )   
            tplr.logger.info(f"Pages: {[p[1] for p in pages]} for UID: {step_uid} and Window: {step_window}")
            
            # Accumulate gradient.
            tplr.logger.info("Start accumulating...")
            self.optimizer.zero_grad()
            self.model.zero_grad()
            total_loss = 0
            for i, batch in enumerate(loader):
                input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                with torch.amp.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
                outputs.loss.backward()
                print('loss:', outputs.loss.item())
                if self.current_window != step_window:
                    break
            tplr.logger.info(f"Stopped accumulating: {i+1} batches with {(i+1) * self.hparams.batch_size * self.hparams.sequence_length} tokens ")
            # Log to wandb.
            if self.config.use_wandb:
                self.wandb.log({"loss": outputs.loss.item()})
                
            # Reduce gradient using DeMo.
            gradient = {}
            xshapes = {}
            totalks = {}
            transmitted = {}
            for n, p in self.model.named_parameters():
                # Step-Weight decay
                p.data.mul_( 1.0 - self.scheduler.get_last_lr()[0] * self.hparams.weight_decay )
                # Momentum decay
                self.momentum[n].mul_( self.hparams.momentum_decay )
                # Add the grad to the momentum.
                self.momentum[n].add_( p.grad, alpha=self.scheduler.get_last_lr()[0] )
                # Compress gradient.
                idxs, vals, xshape, totalk = self.compressor.compress(
                    self.transformer.encode(self.momentum[n]), self.hparams.topk_compression
                )
                # Estimate transmitted gradient.
                transmit_grad = self.transformer.decode(
                    self.compressor.decompress(p, idxs, vals, xshape, totalk)
                )
                # Remove the transmitted from delta (double counting)
                self.momentum[n].sub_(transmit_grad)
                # Add to share_state
                transmitted[ n ] = transmit_grad
                gradient[ n + 'idxs'] = idxs 
                gradient[ n + 'vals'] = vals
                xshapes[ n ] = xshape
                totalks[ n ] = totalk

            # All-gather share state from all peers with timeout.
            tplr.logger.info(f"Start gather: {self.peers}")
            gather_result = await self.comms.gather(
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
                    # Decompress their gradient.
                    their_grad = self.transformer.decode(
                        self.compressor.decompress(p, eval_idx, eval_val, xshapes[ n ], totalks[ n ])
                    )
                    # Get my recreated gradient.
                    my_grad = transmitted[ n ]
                    # Compute cosine sim score.
                    score = torch.nn.functional.cosine_similarity(their_grad.flatten(), my_grad.flatten(), dim=0)
                    # Compute moving scores and weights.
                    self.scores[step_uid] = self.hparams.scores_alpha * score + (1 - self.hparams.scores_alpha) * self.scores[step_uid].expand_as(score)
                    self.weights = torch.softmax(self.scores, dim=0)
                    # Log scores and weights to wandb.
                    if self.config.use_wandb:
                        for uid in self.peers:
                            self.wandb.log({f"s{uid}": self.scores[uid], f"w{uid}": self.weights[uid] })
                    
                # Decompress all gradients in batch form to produce shared gradient.
                new_grad = self.transformer.decode(
                    self.compressor.batch_decompress(
                        p, gather_result[n + 'idxs'], gather_result[n + 'vals'], xshapes[ n ], totalks[ n ]
                    )
                )
                # Set recomputed gathered gradient.
                if p.grad is None:
                    p.grad = new_grad
                else:
                    p.grad.copy_(new_grad)
                # Sign-SGD
                p.grad.sign_()
                    
            # Apply the optimizer step
            tplr.logger.info("Finish and step.")
            self.optimizer.step()
            self.scheduler.step()
            # Set weights on the chain based on current weights.
            if self.config.is_validator and step_window % self.hparams.windows_per_weights == 0:
                
                # Set weights on chain.
                self.subtensor.set_weights(
                    wallet = self.wallet,
                    netuid = self.config.netuid,
                    uids = self.metagraph.uids,
                    weights = self.weights,
                    wait_for_inclusion = False, # Dont wait, fire and forget.
                    wait_for_finalization = False,
                )
                
            
            # Wait for end of window (if not already done.)
            while self.current_window == step_window:
                time.sleep(0.1)
            
    # Listens for new blocks and sets self.current_block and self.current_window
    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event['header']['number'])
            if int( self.current_block / self.hparams.blocks_per_window ) != self.current_window:
                self.current_window = int( self.current_block / self.hparams.blocks_per_window ) 
        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler)
                break
            except Exception:
                time.sleep(1) 

# Start miner/validator.
if __name__ == "__main__":
    asyncio.run( Neuron().run() )
