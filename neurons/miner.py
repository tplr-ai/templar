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
import sys
import time
import random
import asyncio
import argparse
import threading

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
        parser.add_argument('--project', type=str, default='templar', help='Wandb project.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
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
            total_iters=10,
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10000,
            T_mult=2,
            eta_min=self.hparams.learning_rate * 0.1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[10],
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

        self.bucket = self.comms.get_own_bucket()
        self.comms.try_commit(self.wallet, self.bucket)
        self.comms.fetch_commitments()


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
        self.step_counter = 0

        # Add step tracking
        self.global_step = 0
        self.window_step = 0
        
        # Track additional metrics
        self.total_tokens_processed = 0
        self.batch_times = []  # For tracking processing speed
        
        # Initialize WandB
        self.wandb = tplr.initialize_wandb(
            run_prefix='M',
            uid=self.uid,
            config=self.config,
            group='miner',
            job_type='mining'
        )

    # Main training loop.
    async def run(self):
        # Try to load latest checkpoint
        validator_uid, stake = self.comms.get_highest_stake_validator()
        if stake > 0:
            try:
                # Calculate the most recent window that should have a checkpoint
                expected_checkpoint_window = (self.current_window // self.hparams.checkpoint_frequency) * self.hparams.checkpoint_frequency

                # Try last few windows in case of missed checkpoints
                for window in range(expected_checkpoint_window, max(0, expected_checkpoint_window - 3 * self.hparams.checkpoint_frequency), -self.hparams.checkpoint_frequency):
                    result = await self.comms.get(
                        uid=str(validator_uid),
                        window=window,
                        key='checkpoint',
                        timeout=240,
                        local=False,
                        stale_retention=10
                    )
                    if result is None:
                        tplr.logger.debug(f"No checkpoint found for window {window}")
                        continue

                    checkpoint_data, global_step = result
                    try:
                        # Load state dicts from dictionary
                        self.model.load_state_dict(checkpoint_data['model_state_dict'])
                        self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                        self.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                        self.momentum = checkpoint_data['momentum']
                        self.global_step = checkpoint_data['global_step']
                        
                        # Update optimizer and scheduler steps to match
                        self.optimizer._step_count = self.global_step
                        self.scheduler.last_epoch = self.global_step
                        
                        tplr.logger.info(f"Loaded checkpoint from validator {validator_uid} at window {window}, global_step={self.global_step}")
                        break  # Successfully loaded checkpoint, exit loop
                    except KeyError as e:
                        tplr.logger.error(f"Invalid checkpoint format: missing key {e}")
                    except Exception as e:
                        tplr.logger.error(f"Failed to load checkpoint: {e}")
                else:
                    tplr.logger.info("No valid checkpoints found in recent windows")
            except Exception as e:
                tplr.logger.warning(f"Failed to load checkpoint: {e}")
        else:
            tplr.logger.info("No active validators found, starting from scratch")
            self.global_step = 0

        # Start background block listener
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(
            target=self.block_listener,
            args=(self.loop,),
            daemon=True,
        ).start()
        self.comms.start_commitment_fetcher()

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
            tplr.logger.info(f"Pages: {[p[1] for p in pages]} for  Window: {step_window}")
            
            # Accumulate gradient
            start_time = time.time()
            tplr.logger.info("Start accumulating...")
            self.optimizer.zero_grad()
            self.model.zero_grad()
            total_loss = 0
            batch_tokens = 0
            
            for i, batch in enumerate(loader):
                input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                
                with torch.amp.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                
                total_loss += outputs.loss.item()
                outputs.loss.backward()
                
                # Track tokens
                batch_tokens += (labels != -100).sum().item()
                
                tplr.logger.info(f'loss: {outputs.loss.item()}')
                if self.current_window != step_window:
                    tplr.logger.info('<Exhausted window>')
                    break
            tplr.logger.info(f"Stopped accumulating: {i+1} batches with {(i+1) * self.hparams.batch_size * self.hparams.sequence_length} tokens")

            # Calculate processing metrics
            duration = time.time() - start_time
            self.batch_times.append(duration)
            self.total_tokens_processed += batch_tokens

            # Log gradient metrics
            grad_norms = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
            weight_norms = [p.norm().item() for p in self.model.parameters()]
            momentum_norms = [m.norm().item() for m in self.momentum.values()]

            # Enhanced wandb logging with all metrics
            self.wandb.log({
                # Training metrics
                "miner/loss": total_loss/(i+1),
                "miner/tokens_per_sec": ((i+1) * self.hparams.batch_size * self.hparams.sequence_length)/duration,
                "miner/batch_duration": duration,
                "miner/total_tokens": self.total_tokens_processed,
                "miner/batch_tokens": batch_tokens,
                "miner/global_step": self.global_step,
                
                # Resource metrics
                "miner/gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
                "miner/gpu_memory_cached": torch.cuda.memory_reserved() / 1024**2,  # MB
                
                # Network metrics
                "miner/active_peers": len(self.peers),
                "miner/effective_batch_size": len(self.peers) * self.hparams.batch_size,
                
                # Optimization metrics
                "miner/learning_rate": self.scheduler.get_last_lr()[0],
                
                # Gradient statistics as points
                "miner/mean_grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else 0,
                "miner/max_grad_norm": max(grad_norms) if grad_norms else 0,
                "miner/min_grad_norm": min(grad_norms) if grad_norms else 0,
                "miner/grad_norm_std": torch.tensor(grad_norms).std().item() if grad_norms else 0,
                "miner/mean_weight_norm": sum(weight_norms) / len(weight_norms),
                "miner/mean_momentum_norm": sum(momentum_norms) / len(momentum_norms),
            }, step=self.global_step)

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

            # Gather gradients from peers
            tplr.logger.info(f"Start gather: {self.peers}")
            gather_result = await self.comms.gather(
                state_dict=gradient,
                my_uid=self.uid,
                uids=self.peers,
                window=step_window,
                key='gradient',
                timeout=20,
                device=self.config.device,
                local=False,
                stale_retention=10,
                global_step=self.global_step,
            )

            if gather_result is None:
                tplr.logger.error("Failed to gather gradients from peers. Waiting for next window.")
                # Wait for next window
                while self.current_window == step_window:
                    await asyncio.sleep(0.1)
                continue  # Proceed to the next window

            # Update self.global_step based on the maximum global_step received
            max_global_step = max(gather_result.global_steps + [self.global_step])
            if max_global_step > self.global_step:
                tplr.logger.info(f"Updating global_step from {self.global_step} to {max_global_step}")
                self.global_step = max_global_step
                # Update optimizer and scheduler steps
                self.optimizer._step_count = self.global_step
                self.scheduler.last_epoch = self.global_step

            # Decompress state and apply to grad.
            for n, p in self.model.named_parameters():
                idxs_key = n + 'idxs'
                vals_key = n + 'vals'
                idxs = getattr(gather_result.state_dict, idxs_key, None)
                vals = getattr(gather_result.state_dict, vals_key, None)
                if idxs is not None and vals is not None:
                    # Ensure idx and val are lists of tensors
                    if not isinstance(idxs, (list, tuple)):
                        idxs = [idxs]
                    if not isinstance(vals, (list, tuple)):
                        vals = [vals]
                    
                    new_grad = self.transformer.decode(
                        self.compressor.batch_decompress(
                            p.to(self.config.device),
                            idxs,
                            vals,
                            xshapes[n],
                            totalks[n]
                        )
                    )
                    # Set recomputed gathered gradient.
                    if p.grad is None:
                        p.grad = new_grad
                    else:
                        p.grad.copy_(new_grad)
                    # Sign-SGD
                    p.grad.sign_()
                else:
                    tplr.logger.info(f"Gradient data missing for parameter {n}, skipping.")

            # Apply optimizer step
            tplr.logger.info("Finish and step.")
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            self.window_step += 1
            tplr.logger.info(f"Total optimization steps: {self.global_step}")

            # Wait for next window
            tplr.logger.info("Wait for next window...")
            while self.current_window == step_window:
                await asyncio.sleep(0.1)
            self.window_step = 0

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
