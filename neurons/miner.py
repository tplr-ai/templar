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

# Standard library
import os
import sys
import time
import random
import asyncio
import argparse
import threading
from typing import Dict

# Third party
import torch
import numpy as np
import bittensor as bt
from torch.optim import SGD
from transformers import LlamaForCausalLM
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)

# Local
import tplr


# GPU optimizations
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Miner:
    
    # Command line config items.
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Miner script')
        parser.add_argument('--netuid', type=int, default=268, help='Bittensor network UID.')
        parser.add_argument('--project', type=str, default='llama-demo-1', help='Wandb project.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
        parser.add_argument('--peers', type=int, nargs='+', default=[], help='List of UIDs to peer with')
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        config = bt.config(parser)
        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()
        return config
    
    def __init__(self):
        tplr.logger.debug("Starting initialization...")
        
        # Init config and load hparams
        self.config = Miner.config()
        self.hparams = tplr.load_hparams()
        
        # Init bittensor objects
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            tplr.logger.error(f'\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}[/bold]')
            sys.exit()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        
        # Init model with hparams config
        self.model = LlamaForCausalLM(self.hparams.model_config)
        self.model.to(self.config.device)
        self.tokenizer = self.hparams.tokenizer
        
        # Init optimizer and momentum
        self.optimizer = SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        self.momentum = {}
        for n, p in self.model.named_parameters():
            self.momentum[n] = torch.zeros_like(p)
        
        # Set up scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=250,
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
            eta_min=self.hparams.learning_rate * 0.1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[250],
        )

        # Init compression
        self.transformer = tplr.compress.TransformDCT(
            self.model,
            target_chunk=self.hparams.target_chunk,
        )
        self.compressor = tplr.compress.CompressDCT()

        # Init comms
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location='/tmp',
            key_prefix='model',
            config=self.config,
            netuid=self.config.netuid,
            metagraph=self.metagraph,
            hparams=self.hparams,
        )

        # Init peers
        if not self.config.peers:
            self.peers = self.comms.peers
            tplr.logger.info(f'Filtered peers with buckets: {self.peers}')
        else:
            self.peers = self.config.peers
        if self.uid not in self.peers:
            self.peers.append(self.uid)

        # Init state params
        self.stop_event = asyncio.Event()
        self.current_block = self.subtensor.block
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)

        # Init wandb
        if self.config.use_wandb:
            self.wandb = tplr.WandbManager(
                uid=self.uid,
                config=self.config,
                is_validator=False,
            ).run

    # Main training loop.
    async def run(self):
        # Try to load latest checkpoint
        validator_uid, stake = self.comms.get_highest_stake_validator()
        if stake > 0:
            try:
                state_dict = await self.comms.get(
                    uid=str(validator_uid),
                    window=self.current_window,
                    key='checkpoint',
                    timeout=240
                )
                if state_dict is not None:
                    self.model.load_state_dict(state_dict)
                    tplr.logger.info(f"Loaded checkpoint from validator {validator_uid} at window {self.current_window}")
                else:
                    tplr.logger.info("No checkpoint found, starting from scratch")
            except Exception as e:
                tplr.logger.warning(f"Failed to load checkpoint: {e}")
        else:
            tplr.logger.info("No active validators found, starting from scratch")

        # Start background block listener
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(
            target=self.block_listener,
            args=(self.loop,),
            daemon=True,
        ).start()

        while True:
            step_window = self.current_window
            tplr.logger.info(f"\n{'-' * 40} Window: {step_window} {'-' * 40}")

            # Get the pages for this window.
            pages = await tplr.dataset.DatasetLoader.next_pages(
                offset = step_window,
                n_pages = self.hparams.pages_per_window,
                seed = self.metagraph.hotkeys[self.uid]
            )            
            loader = await tplr.dataset.DatasetLoader.create(
                batch_size = self.hparams.batch_size,
                sequence_length = self.hparams.sequence_length,
                pages_info = pages,
                tokenizer = self.tokenizer
            )   
            tplr.logger.info(f"Pages: {[p[1] for p in pages]} for UID: {self.config.uid} and Window: {step_window}")
            
            # Accumulate gradient.
            start_time = time.time()
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
                tplr.logger.info(f'loss: {outputs.loss.item()}')
                if self.current_window != step_window:
                    tplr.logger.info('<Exhausted window>')
                    break
            tplr.logger.info(f"Stopped accumulating: {i+1} batches with {(i+1) * self.hparams.batch_size * self.hparams.sequence_length} tokens")
            duration = time.time() - start_time
            
            # Log metrics
            if self.config.use_wandb:
                self.wandb.log({
                    "loss": total_loss/(i+1),
                    "tokens_per_sec": ((i+1) * self.hparams.batch_size * self.hparams.sequence_length)/duration
                })
            
            # Reduce gradient using DeMo.
            gradient = {}
            xshapes = {}
            totalks = {}
            transmitted = {}
            for n, p in self.model.named_parameters():
                # Step-Weight decay
                p.data.mul_(1.0 - self.scheduler.get_last_lr()[0] * self.hparams.weight_decay)
                # Momentum decay
                self.momentum[n].mul_(self.hparams.momentum_decay)
                # Add the grad to the momentum
                self.momentum[n].add_(p.grad, alpha=self.scheduler.get_last_lr()[0])
                # Compress gradient
                idxs, vals, xshape, totalk = self.compressor.compress(
                    self.transformer.encode(self.momentum[n]), 
                    self.hparams.topk_compression
                )
                # Estimate transmitted gradient
                transmit_grad = self.transformer.decode(
                    self.compressor.decompress(p, idxs, vals, xshape, totalk)
                )
                # Remove transmitted from delta
                self.momentum[n].sub_(transmit_grad)
                # Add to share_state
                transmitted[n] = transmit_grad
                gradient[n + 'idxs'] = idxs
                gradient[n + 'vals'] = vals
                xshapes[n] = xshape
                totalks[n] = totalk

            # All-gather share state from peers
            tplr.logger.info(f"Start gather: {self.peers}")
            gather_result = await self.comms.gather(
                state_dict=gradient,
                my_uid=self.uid,
                uids=self.peers,
                window=step_window,
                key='gradient',
                timeout=5,
                device=self.config.device
            )
            
            # Decompress state and apply grad
            for n, p in self.model.named_parameters():
                # Decompress all gradients in batch form
                new_grad = self.transformer.decode(
                    self.compressor.batch_decompress(
                        p, gather_result[n + 'idxs'], gather_result[n + 'vals'], 
                        xshapes[n], totalks[n]
                    )
                )
                # Set recomputed gathered gradient
                if p.grad is None:
                    p.grad = new_grad
                else:
                    p.grad.copy_(new_grad)
                # Sign-SGD
                p.grad.sign_()
                
            # Apply optimizer step
            tplr.logger.info("Finish and step.")
            self.optimizer.step()
            self.scheduler.step()
            if self.config.use_wandb:
                self.wandb.log({"lr": self.scheduler.get_last_lr()[0]})

            # Wait for end of window
            tplr.logger.info("Wait for next window...")
            while self.current_window == step_window:
                await asyncio.sleep(0.1)

    # Listens for new blocks and sets self.current_block and self.current_window
    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event['header']['number'])
            if int(self.current_block / self.hparams.blocks_per_window) != self.current_window:
                self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler)
                break
            except Exception:
                time.sleep(1)

# Start miner/validator.
if __name__ == "__main__":
    asyncio.run(Miner().run())
