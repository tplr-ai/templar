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

# GPU optimizations.
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Validator:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Validator script')
        parser.add_argument('--netuid', type=int, default=268, help='Bittensor network UID.')
        parser.add_argument('--project', type=str, default='templar', help='Wandb project.')
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
        self.config = Validator.config()
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
        
        # Init compression
        self.transformer = tplr.compress.TransformDCT(
            self.model, 
            target_chunk=self.hparams.target_chunk
        )
        self.compressor = tplr.compress.CompressDCT()
        
        # Init optimizer and momentum
        self.optimizer = SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        self.momentum = {}
        self.xshapes = {}
        self.totalks = {}
        for n, p in self.model.named_parameters():
            self.momentum[n] = torch.zeros_like(p)
            _, _, xshape, totalk = self.compressor.compress(
                self.transformer.encode(self.momentum[n]), 
                self.hparams.topk_compression
            )
            self.xshapes[n] = xshape
            self.totalks[n] = totalk

        # Set up scheduler setup
        warmup_scheduler = LinearLR(
            self.optimizer,
            total_iters=250
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10000,
            T_mult=1,
            eta_min=self.hparams.learning_rate * 0.1
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[250]
        )

        # Init comms with required chain management args
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location='/tmp',
            key_prefix='model',
            config=self.config,
            netuid=self.config.netuid,
            metagraph=self.metagraph,
            hparams=self.hparams,
        )

        self.comms.setup()
        
        # Init peers
        if not self.config.peers:
            self.peers = self.comms.peers
            tplr.logger.info(f'Filtered peers with buckets: {self.peers}')
        else:
            self.peers = self.config.peers

        # Init state params
        self.stop_event = asyncio.Event()
        self.current_block = self.subtensor.block
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        self.sync_window = self.current_window

        # Init scores
        self.scores = torch.zeros(self.metagraph.n, dtype=torch.float32)
        self.moving_avg_scores = torch.zeros(self.metagraph.n, dtype=torch.float32) 
        self.ma_alpha = 0.95  # Moving average decay factor

        # Add step tracking
        self.global_step = 0
        self.window_step = 0
        self.eval_count = 0  # Track number of evaluations
        
        # Initialize WandB
        self.wandb = tplr.initialize_wandb(
            run_prefix='V',
            uid=self.uid,
            config=self.config,
            group='validator',
            job_type='validation'
        )

    async def run(self):
        # Try to load latest checkpoint
        validator_uid, stake = self.comms.get_highest_stake_validator()
        if stake > 0:
            try:
                state_dict = await self.comms.get(
                    uid=str(validator_uid),
                    window=self.current_window,
                    key='checkpoint',
                    timeout=240,
                    local=False
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
        # Start block listener
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(
            target=self.block_listener, 
            args=(self.loop,), 
            daemon=True
        ).start()

        while True:
            step_window = self.current_window
            # Wait for validator offset
            while self.sync_window >= (self.current_window - self.hparams.validator_offset):
                tplr.logger.info(f'Waiting for validator window offset, synced: {self.sync_window}, current:{self.current_window}, offset:{self.hparams.validator_offset}')
                await asyncio.sleep(12)

            # Check if checkpointing is needed (every 500 windows)
            if self.current_window % 500 == 0:
                tplr.logger.info(f'Creating checkpoint at window {self.current_window}')
                
                try:
                    # Upload the model state directly using put
                    await self.comms.put(
                        state_dict=self.model.state_dict(),
                        uid=self.uid,
                        window=self.current_window,
                        key='checkpoint',
                        local=False
                    )
                    tplr.logger.info(f"Successfully created checkpoint at window {self.current_window}")
                except Exception as e:
                    tplr.logger.error(f"Failed to create checkpoint: {e}")

            # Log checkpoint creation
            if self.global_step % 500 == 0:
                self.wandb.log({
                    "checkpoint_window": self.current_window,
                    "global_step": self.global_step,
                }, step=self.global_step)

            # Catch up to current - validator_offset
            while self.sync_window < (self.current_window - self.hparams.validator_offset):
                self.sync_window += 1
                tplr.logger.info(f'Syncing window: {self.sync_window} current: {self.current_window}')

                # Gather gradients from this window
                step_grads = await self.comms.gather(
                    state_dict={},
                    my_uid=self.uid,
                    uids=self.peers,
                    window=self.sync_window,
                    key='gradient',
                    timeout=5,
                    device=self.config.device,
                    local=False
                )

                # Check if any gradients were gathered
                if step_grads is None:
                    tplr.logger.info("No gradients received, waiting for next window.")
                    continue

                tplr.logger.info(f"Received gradients from UIDs: {step_grads.uids}")

                # Decompress state and apply to gradients
                for n, p in self.model.named_parameters():
                    # Initialize an empty tensor for the aggregated gradient
                    aggregated_grad = torch.zeros_like(p, device=self.config.device)

                    # Sum gradients from all valid UIDs
                    for idx, uid in enumerate(step_grads.uids):
                        new_grad = self.transformer.decode(
                            self.compressor.decompress(
                                p.to(self.config.device),
                                step_grads.state_dict.__dict__[n + 'idxs'][idx],
                                step_grads.state_dict.__dict__[n + 'vals'][idx],
                                self.xshapes[n], self.totalks[n]
                            )
                        )

                        # Aggregate the gradients (e.g., sum or average)
                        # Here, we'll sum them up
                        aggregated_grad.add_(new_grad)

                    # Optionally average the gradient
                    aggregated_grad.div_(len(step_grads.uids))

                    # Set the aggregated gradient
                    if p.grad is None:
                        p.grad = aggregated_grad
                    else:
                        p.grad.copy_(aggregated_grad)
                    p.grad.sign_()

                # Apply the optimizer step
                self.optimizer.step()
                self.scheduler.step()

                self.wandb.log({"lr": self.scheduler.get_last_lr()[0]}, step=self.global_step)
                
            # Get a random peer to eval on their gradient at self.sync_window + 1
            eval_uid = random.choice(step_grads.uids)
            # Get the pages for the window infront of the current sync window
            pages = await tplr.dataset.DatasetLoader.next_pages(
                offset=self.sync_window + 1,
                n_pages=self.hparams.pages_per_window,
                seed=eval_uid
            )            
            loader = await tplr.dataset.DatasetLoader.create(
                batch_size=self.hparams.batch_size,
                sequence_length=self.hparams.sequence_length,
                pages_info=pages,
                tokenizer=self.tokenizer
            )   
            tplr.logger.info(f'Evaluating uid: {eval_uid} on window: {self.sync_window + 1} with state from: {self.sync_window} and pages: {[p[1] for p in pages]}')
            
            # Compute and log loss before gradient application
            loss_before = 0
            n_tokens = 0
            for i, batch in enumerate(loader):
                input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                loss_before += self.model(input_ids=input_ids, labels=labels).loss.item()
                n_tokens += (labels != -100).sum().item()
            
            loss_before_per_token = loss_before / n_tokens if n_tokens > 0 else 0
            tplr.logger.info(f'Computed total loss before: {loss_before} ({loss_before_per_token:.4f} per token)')

            # Get the gradients from this miner on this window
            eval_grad = await self.comms.get(
                uid=eval_uid,
                window=self.sync_window + 1,
                key='gradient',
                timeout=5,
                local=False,
                stale_retention=10
            )
            if eval_grad is None:
                score = 0
                tplr.logger.info(f'Miner with uid: {eval_uid} has no gradient for window: {self.sync_window + 1}')
                continue

            # Apply grad to model which is at state sync_window
            for n, p in self.model.named_parameters():  
                # Decompress their gradient
                decompressed_grad = self.transformer.decode( 
                    self.compressor.decompress(
                        p.to(self.config.device),
                        eval_grad[n + 'idxs'].to(self.config.device), 
                        eval_grad[n + 'vals'].to(self.config.device),
                        self.xshapes[n], self.totalks[n],
                    )
                )
                # Apply this grad to the param of the model using the learning rate of the scheduler
                p.data.sub_(decompressed_grad, alpha=self.scheduler.get_last_lr()[0]) 
                
            # Compute loss after gradient application
            loss_after = 0
            n_tokens = 0
            for i, batch in enumerate(loader):
                input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                loss_after += self.model(input_ids=input_ids, labels=labels).loss.item()
                n_tokens += (labels != -100).sum().item()
                if self.current_window != step_window:
                    tplr.logger.info('<Exhausted window>')
                    break
            
            loss_after_per_token = loss_after / n_tokens if n_tokens > 0 else 0
            tplr.logger.info(f'Computed total loss after: {loss_after} ({loss_after_per_token:.4f} per token)')
     
            # Remove gradient from the model
            for n, p in self.model.named_parameters():  
                # Decompress their gradient
                decompressed_grad = self.transformer.decode( 
                    self.compressor.decompress(
                        p.to(self.config.device),
                        eval_grad[n + 'idxs'].to(self.config.device), 
                        eval_grad[n + 'vals'].to(self.config.device),
                        self.xshapes[n], self.totalks[n],
                    )
                )
                # Apply this grad to the param of the model using the learning rate of the scheduler
                p.data.add_(decompressed_grad, alpha=self.scheduler.get_last_lr()[0]) 
                
            # Compute improvement metrics
            loss_improvement = loss_before - loss_after
            improvement_percentage = ((loss_before - loss_after) / loss_before * 100) if loss_before != 0 else 0

            # Compute score
            score = loss_before - loss_after
            tplr.logger.info(f'score: {score}, loss_before: {loss_before_per_token:.4f}, loss_after: {loss_after_per_token:.4f}, loss_improvement: {loss_improvement:.4f}, improvement_percentage: {improvement_percentage:.2f}%, uid: {eval_uid}')

            # Log comprehensive metrics
            self.wandb.log({
                "validator/loss_before": loss_before_per_token,
                "validator/loss_after": loss_after_per_token,
                "validator/loss_improvement": loss_improvement,
                "validator/improvement_percentage": improvement_percentage,
                "validator/eval_count": self.eval_count,
                "validator/tokens_evaluated": n_tokens,
                "validator/learning_rate": self.scheduler.get_last_lr()[0],
            }, step=self.global_step)

            # Update counters
            self.global_step += 1
            self.eval_count += 1

            # Update scores with new score
            self.scores[eval_uid] = self.hparams.scores_alpha * score + (1 - self.hparams.scores_alpha) * self.scores[eval_uid]
            # Update moving average scores
            self.moving_avg_scores[eval_uid] = self.ma_alpha * self.moving_avg_scores[eval_uid] + (1 - self.ma_alpha) * score
            # Compute weights from moving average scores
            # Zero out negative scores and apply softmax only on positive scores
            positive_scores = torch.where(self.moving_avg_scores > 0, self.moving_avg_scores, torch.zeros_like(self.moving_avg_scores))
            weights = positive_scores / positive_scores.sum() if positive_scores.sum() > 0 else torch.zeros_like(positive_scores)

            # Log per-UID metrics
            valid_score_indices = torch.nonzero(self.scores > 0).squeeze().view(-1)
            for uid_i in valid_score_indices:
                uid = uid_i.item()
                self.wandb.log({
                    f"validator/scores/{uid}": self.scores[uid_i].item(),
                    f"validator/moving_avg_scores/{uid}": self.moving_avg_scores[uid_i].item(),
                    f"validator/weights/{uid}": weights[uid_i].item(),
                }, step=self.global_step)

            # Log aggregate network statistics
            self.wandb.log({
                "validator/active_miners": len(valid_score_indices),
                "validator/mean_score": self.scores[valid_score_indices].mean().item(),
                "validator/mean_moving_avg_score": self.moving_avg_scores[valid_score_indices].mean().item(),
                "validator/max_score": self.scores.max().item(),
                "validator/min_score": self.scores.min().item(),
                "validator/max_moving_avg_score": self.moving_avg_scores.max().item(),
                "validator/min_moving_avg_score": self.moving_avg_scores.min().item(),
                "validator/mean_weight": weights[valid_score_indices].mean().item(),
                "validator/weight_std": weights[valid_score_indices].std().item(),
                "validator/score_std": self.scores[valid_score_indices].std().item(),
                "validator/moving_avg_score_std": self.moving_avg_scores[valid_score_indices].std().item(),
                "validator/max_weight": weights.max().item(),
                "validator/min_weight": weights.min().item(),
            }, step=self.global_step)


            if self.sync_window % self.hparams.windows_per_weights == 0:
                # Set weights on chain
                self.subtensor.set_weights(
                    wallet=self.wallet,
                    netuid=self.config.netuid,
                    uids=self.metagraph.uids,
                    weights=weights,
                    wait_for_inclusion=False,
                    wait_for_finalization=False,
                )
                tplr.logger.info(f'Set weights on chain for window {self.sync_window}')



            # Apply the optimizer step
            tplr.logger.info("Finish and step.")
            self.optimizer.step()
            self.scheduler.step()
            tplr.logger.info(f"Total optimization steps: {self.global_step}")

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

if __name__ == "__main__":
    asyncio.run(Validator().run())
