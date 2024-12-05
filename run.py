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

import tplr

class Miner:
    
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Miner script')
        parser.add_argument('--project', type=str, default='templar', help='Optional wandb project name')
        parser.add_argument('--netuid', type=int, default=229, help='Bittensor network UID.')
        parser.add_argument('--windows_per_sync', type=int, default=10, help='Windows between state sync between nodes.')
        parser.add_argument('--batch_size', type=int, default=8, help='Training batch size per accumulation.')
        parser.add_argument('--sequence_length', type=int, default=1024, help='Training batch size per accumulation.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
        parser.add_argument('--is_validator', action='store_true', help='Is validator')
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        bt.logging.add_args( parser )
        config = bt.config(parser)
        if config.debug: tplr.debug()
        if config.trace: tplr.trace()
        return config
    
    def __init__(self):
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

        # Initialize the model with random weights.
        model_config = GPT2LMHeadModel.config_class()
        self.model = GPT2LMHeadModel(model_config)
        self.model.to(self.config.device)
        self.model.init_weights()
        
        # Init tokenizer.
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Init optimizer.
        self.momentum = {}
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001)          
        for n, p in self.model.named_parameters():
            self.momentum[n] = torch.zeros_like(p)
        
        # Init compression.
        self.transformer = tplr.compress.TransformDCT( self.model, target_chunk = 64 )
        self.compressor = tplr.compress.CompressDCT()
        
        # Init state params.
        self.stop_event = asyncio.Event()
        self.current_window = int( self.subtensor.block / 2 )
        
        # Init scores
        self.scores = torch.zeros( [0] * int(self.metagraph.n), dtype = torch.float32 )
        
        # Init wandb.
        if self.config.use_wandb:
            # Delete all runs with my name and create a new one.
            try:
                for run in wandb.Api().runs(path=self.config.project):
                    if run.name == f'M{self.uid}':
                        tplr.logger.info(f'Deleting old run: {run}'); run.delete()
            except: pass
            wandb.init(project=self.config.project, resume='allow', name=f'M{self.uid}', config=self.config)
        
    async def run( self ):

        # Get pages.        
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(target=self.block_listener, args=(self.loop,), daemon=True).start()
        
        gather_result = await tplr.comms.gather(
            state_dict = self.model.state_dict(),
            my_uid = self.uid,
            uids = [2,3],
            window = int(self.current_window/self.config.windows_per_sync),
            key = 'model',
            timeout = 30,
            device = self.config.device
        )
        state_dict = {}
        for name in gather_result:
            tensor_list = gather_result[name]
            averaged_tensor = torch.mean(torch.stack(tensor_list), dim=0)
            state_dict[name] = averaged_tensor
        self.model.load_state_dict(state_dict)

        # Run until stopped.
        while True:
            
            # Record the window we are on.
            step_window = self.current_window
            step_uid = self.uid if not self.config.is_validator else random.choice([2,3])
            tplr.logger.info('\n' + '-' * 40 + f' Window: {step_window} ' + '-' * 40)

            # Get the pages for this window.
            pages = await tplr.dataset.DatasetLoader.next_pages(
                offset = step_window,
                n_pages = 1,
                seed = self.metagraph.hotkeys[self.uid]
            )            
            loader = await tplr.dataset.DatasetLoader.create(
                batch_size = self.config.batch_size,
                sequence_length = self.config.sequence_length,
                pages_info = pages,
                tokenizer = self.tokenizer
            )   
            tplr.logger.info(f"Pages: {[p[1] for p in pages]}")
            
            # Accumulate gradient.
            tplr.logger.info(f"Start accumulating...")
            self.optimizer.zero_grad()
            self.model.zero_grad()
            for _, batch in enumerate( loader ):
                input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                outputs = self.model(input_ids=input_ids, labels=labels)
                outputs.loss.backward()
                print ('loss:', outputs.loss.item())
                if self.current_window != step_window:
                    tplr.logger.info(f"Stopped accumulating {self.current_window} != {step_window}")
                    break
            if self.config.use_wandb: wandb.log({f"loss": outputs.loss.item()})
                
            # Create gradient share state.
            share_state = {}
            xshapes = {}
            totalks = {}
            transmitted = {}
            for n, p in self.model.named_parameters():
                # Momentum decay
                self.momentum[n].mul_(0.999)
                # Add the grad to the momentum.
                self.momentum[n].add_(p.grad, alpha=0.0001)
                
                # Compress gradient.
                sparse_idx, sparse_val, xshape, totalk = self.compressor.compress(
                    self.transformer.encode(self.momentum[n]), 32
                )
                # Estimate transmitted gradient.
                transmit_grad = self.transformer.decode(
                    self.compressor.decompress(p, sparse_idx, sparse_val, xshape, totalk)
                )
                # Remove the transmitted from delta (double couting)
                self.momentum[n].sub_(transmit_grad)
                # Add to share_state
                transmitted[ n ] = transmit_grad
                share_state[ n + 'sparse_idx'] = sparse_idx 
                share_state[ n + 'sparse_val'] = sparse_val
                xshapes[ n ] = xshape; totalks[ n ] = totalk

            # All-gather share state from all peers with timeout.
            gather_result = await tplr.comms.gather(
                state_dict = share_state,
                my_uid = self.uid,
                uids = [2,3],
                window = step_window,
                key = 'gradient',
                timeout = 3,
                device = self.config.device
            )
            
            # Decompress state and apply to grad.
            for n, p in self.model.named_parameters():
                # Decode grad from all nodes
                if self.config.is_validator:
                    eval_idx = gather_result[n + 'sparse_idx'][ step_uid ]
                    eval_val = gather_result[n + 'sparse_val'][ step_uid ]
                    # Estimate transmitted gradient.
                    their_grad = self.transformer.decode(
                        self.compressor.decompress(p, eval_idx, eval_val, xshapes[ n ], totalks[ n ])
                    )
                    # Get transmitted
                    my_grad = transmitted[ n ]
                    # Compute cosine sim
                    cosine_sim = torch.nn.functional.cosine_similarity(their_grad, my_grad, dim=0)
                    self.scores[ step_uid ] = 0.1 * cosine_sim + 0.9 * self.scores[ step_uid ]
                    self.weights = torch.softmax( self.scores, dim = 0 )
                    
                new_grad = self.transformer.decode(
                    self.compressor.batch_decompress(
                        p, gather_result[n + 'sparse_idx'], gather_result[n + 'sparse_val'], xshapes[ n ], totalks[ n ]
                    )
                )
                # Set grad to values
                if p.grad is None:
                    p.grad = new_grad
                else:
                    p.grad.copy_(new_grad)
                # Sign-SGD
                p.grad.sign_()
                    
            # Apply the optimizer step
            self.optimizer.step()
            
            # Set weights.
            if self.config.is_validator:
                
                # Set weights on chain.
                self.subtensor.set_weights(
                    wallet = self.wallet,
                    netuid = self.config.netuid,
                    uids = self.metagraph.uids,
                    weights = self.weights,
                    wait_for_inclusion = True,
                    wait_for_finalization = False,
                )
            
            # Wait for end of window.
            while self.current_window == step_window: time.sleep(0.1)
            
    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event['header']['number'])
            if int( self.current_block / 2 ) != self.current_window:
                self.current_window = int( self.current_block / 2 ) 
        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler); break
            except Exception as e:
                time.sleep(1) 

if __name__ == "__main__":
    asyncio.run( Miner().run() )