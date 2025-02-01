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
#type: ignore

# Standard library
import sys
import time
import random
import asyncio
import argparse
import threading
from contextlib import contextmanager
from time import perf_counter

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

@contextmanager
def timer(name: str, wandb_obj=None, step=None):
    start = perf_counter()
    yield
    duration = perf_counter() - start
    tplr.logger.debug(f"{name} took {duration:.2f}s")
    if wandb_obj and step is not None:
        wandb_obj.log({f"validator/{name}": duration}, step=step)

class Validator:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Validator script')
        parser.add_argument('--netuid', type=int, default=268, help='Bittensor network UID.')
        parser.add_argument('--project', type=str, default='templar', help='Wandb project.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        parser.add_argument('--store-gathers', action='store_true', help='Store gathered gradients in R2')
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
            start_factor=0.1,
            end_factor=1.0,
            total_iters=250,
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10000,
            T_mult=2,
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
            uid=self.uid, 
        )


        self.bucket = self.comms.get_own_bucket('gradients', 'read')
        self.comms.try_commit(self.wallet, self.bucket)
        self.comms.fetch_commitments()
        
        
        # Init state params
        self.stop_event = asyncio.Event()
        self.current_block = self.subtensor.block
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        self.start_window = self.current_window  # Record the start window
        self.global_step = 0  # Initialize global_step to zero
        self.comms.current_window = self.current_window 
        self.sync_window = self.current_window

        # Init scores and tracking
        self.scores = torch.zeros(self.metagraph.n, dtype=torch.float32)
        self.moving_avg_scores = torch.zeros(self.metagraph.n, dtype=torch.float32) 
        self.evaluated_uids = set()  # Track which UIDs we've seen

        # Add step tracking
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

        # Initialize peers
        self.peers = []
        self.eval_peers = []

        # Track inactive peer scores
        self.inactive_scores = {}  # {uid: (last_active_window, last_score)}
        self.inactivity_slash_rate = 0.25  # 25% slash per window

        # binary moving averages
        self.binary_moving_averages = {}

    async def run(self):
        # Load Peers
        if not self.config.peers:
            self.peers = self.comms.peers
            tplr.logger.info(f'Filtered gather peers with buckets: {self.peers}')
        else:
            self.peers = self.config.peers
        if self.uid not in self.peers:
            self.peers.append(self.uid)

        self.comms.commitments = self.comms.get_commitments_sync()
        self.comms.update_peers_with_buckets()
        tplr.logger.info(f"Loaded commitments: {self.comms.commitments.keys()}")

        # Only post start window if you are the highest stake validator
        if (self.uid == self.metagraph.S.argmax().item()):
            # Post start_window to R2
            await self.comms.post_start_window(self.start_window)
            tplr.logger.info(f"This validator is the highest staked. Posted start_window: {self.start_window}")
        else:
            tplr.logger.info("This validator is not the highest staked. Waiting to fetch start_window.")
            # Fetch start_window from highest stake validator
            self.start_window = await self.comms.get_start_window()
            self.global_step = self.current_window - self.start_window
            tplr.logger.info(f"Using start_window: {self.start_window}, global_step: {self.global_step}")

        # Proceed to load checkpoint
        success, loaded_momentum, loaded_global_step, loaded_optimizer, loaded_scheduler = await self.comms.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer, 
            scheduler=self.scheduler,
            transformer=self.transformer,
            compressor=self.compressor,
            current_window=self.current_window,
            device=self.config.device,
            peers=self.peers,
            uid=self.uid
        )
        if success:
            self.momentum = loaded_momentum
            self.global_step = loaded_global_step
            self.optimizer = loaded_optimizer
            self.scheduler = loaded_scheduler
            tplr.logger.info(
                f"Loaded checkpoint with global_step={self.global_step}, "
                f"optimizer_step={self.optimizer.state_dict()['state'].get(0, {}).get('step', 0)}, "
                f"scheduler_step={self.scheduler.last_epoch}"
            )
        else:
            tplr.logger.info("Starting from scratch")
            self.momentum = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
            self.model.to(self.config.device)
        # Start block listener
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(
            target=self.block_listener, 
            args=(self.loop,), 
            daemon=True
        ).start()
        self.comms.start_commitment_fetcher()
        self.comms.start_background_tasks()

        while True:
            # Check for catch-up need
            # catch_up_success, new_global_step, new_optimizer, new_scheduler = await self.comms.check_and_perform_catch_up(
            #     model=self.model,
            #     optimizer=self.optimizer,
            #     scheduler=self.scheduler,
            #     transformer=self.transformer,
            #     compressor=self.compressor,
            #     current_window=self.current_window,
            #     sync_window=self.sync_window,
            #     device=self.config.device,
            #     peers=self.peers,
            #     uid=self.uid,
            #     global_step=self.global_step,
            #     hparams=self.hparams
            # )
            
            # if catch_up_success:
            #     self.global_step = new_global_step
            #     self.optimizer = new_optimizer
            #     self.scheduler = new_scheduler
            #     self.sync_window = self.current_window
            #     continue

            # Normal processing continues...
            while self.sync_window >= (self.current_window - self.hparams.validator_offset):
                tplr.logger.info(f'Waiting for validator window offset, synced: {self.sync_window}, current:{self.current_window}, offset:{self.hparams.validator_offset}')
                await asyncio.sleep(12)

            tplr.logger.info(f'Sync Window: {self.sync_window}, Scheduler epoch: {self.scheduler.last_epoch}, Global step: {self.global_step}')
            
            # 2. Increment sync window and update peer lists
            window_start = tplr.T()
            self.sync_window += 1
            tplr.logger.info(f'Processing window: {self.sync_window} current: {self.current_window}')

            peer_start = tplr.T()
            self.comms.update_peers_with_buckets()
            self.peers = self.comms.peers
            self.eval_peers = self.comms.eval_peers
            tplr.logger.info(f'{tplr.P(self.sync_window, tplr.T() - peer_start)} Updated peers - gather:{len(self.peers)}, eval:{len(self.eval_peers)}')

            tplr.logger.info(f'Current gather peers: {self.peers}')
            tplr.logger.info(f'Current evaluation peers: {self.eval_peers}')

            # After self.comms.update_peers_with_buckets()
            newly_inactive = self.comms.inactive_peers
            current_window = self.sync_window
            
            # Process newly inactive peers
            for uid in newly_inactive:
                if uid not in self.inactive_scores:
                    self.inactive_scores[uid] = (current_window, self.moving_avg_scores[uid].item())
                    tplr.logger.info(f"UID {uid} became inactive at window {current_window} with score {self.moving_avg_scores[uid].item():.4f}")
            
            # Apply penalties to all inactive peers
            for uid, (inactive_since, _) in list(self.inactive_scores.items()):
                # If peer became active again, remove from inactive tracking
                if uid in self.eval_peers:
                    del self.inactive_scores[uid]
                    tplr.logger.info(f"UID {uid} became active again")
                    continue
                
                # Apply flat 25% penalty instead of exponential decay
                old_score = self.moving_avg_scores[uid].item()
                self.moving_avg_scores[uid] *= 0.75  # Apply flat 25% reduction
                new_score = self.moving_avg_scores[uid].item()
                
                tplr.logger.info(
                    f"UID {uid} penalized for inactivity: "
                    f"{old_score:.4f} -> {new_score:.4f}"
                )
                
                # Log slash metrics
                self.wandb.log({
                    f"validator/inactivity/{uid}/score_before": old_score,
                    f"validator/inactivity/{uid}/score_after": new_score,
                }, step=self.global_step)

            # 3. Gather gradients from peers
            gather_start = tplr.T()
            gather_result = await self.comms.gather(
                state_dict=None,
                my_uid=self.uid,
                uids=self.peers,
                window=self.sync_window,
                key='gradient',
                timeout=5,
                device=self.config.device,
                local=False,
                stale_retention=100,
                global_step=self.global_step,
                store_gathers=self.config.store_gathers
            )
            tplr.logger.info(f'{tplr.P(self.sync_window, tplr.T() - gather_start)} Gathered gradients from peers')

            # Add check for empty eval_peers
            if not self.eval_peers:
                tplr.logger.warning(f"No peers available for evaluation in window {self.sync_window}. Waiting for next window.")
                self.global_step += 1
                continue

            # 5. Save original model state for evaluation
            eval_start = tplr.T()
            original_params = {n: p.clone() for n, p in self.model.named_parameters()}

            # 6. Select and evaluate random miner
            eval_uid = random.choice(self.eval_peers)
            tplr.logger.info(f'Evaluating uid: {eval_uid}')

            eval_result = await self.comms.get(
                uid=str(eval_uid),
                window=self.sync_window,
                key='gradient',
                timeout=30,
                local=False,
                stale_retention=10
            )

            scoring_start = tplr.T()
            if eval_result is not None and eval_result[0] is not None:
                # 7. Load evaluation data from own page
                data_start = tplr.T()
                pages = await tplr.r2_dataset.R2DatasetLoader.next_pages(
                    offset=self.sync_window,
                    n_pages=self.hparams.pages_per_window,
                    seed=eval_uid
                )
                loader = await tplr.r2_dataset.R2DatasetLoader.create(
                    batch_size=self.hparams.batch_size,
                    sequence_length=self.hparams.sequence_length,
                    pages_info=pages,
                    tokenizer=self.tokenizer
                )
                tplr.logger.info(f'{tplr.P(self.sync_window, tplr.T() - data_start)} Loaded evaluation data')
                state_dict, _ = eval_result

                # 8. Compute initial loss
                self.optimizer.zero_grad()
                self.model.zero_grad()
                loss_before_own = 0.0
                n_batches = 0

                with torch.no_grad():
                    self.model.eval()
                    # First pass to count batches and store them
                    batches = []
                    for batch in loader:
                        batches.append(batch)
                    
                    total_batches = len(batches)
                    sample_size = max(1, int(total_batches * self.hparams.validator_sample_rate))
                    sampled_indices = random.sample(range(total_batches), sample_size)
                    sampled_indices = sorted(sampled_indices)  # Sort for sequential access
                    
                    tplr.logger.info(f"Evaluating {sample_size}/{total_batches} batches ({self.hparams.validator_sample_rate*100:.1f}%)")
                    
                    for i, batch in enumerate(batches):
                        if i not in sampled_indices:
                            continue
                            
                        input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                        labels = input_ids.clone()
                        labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        loss_before_own += outputs.loss.item()
                        n_batches += 1
                        del input_ids, labels, outputs
                        torch.cuda.empty_cache()

                loss_before_per_batch = loss_before_own / n_batches if n_batches > 0 else 0
                tplr.logger.info(f'Loss before (own data): {loss_before_per_batch}')

                # 9. Apply gradient and compute loss after
                self.optimizer.zero_grad()
                self.model.zero_grad()

                for n, p in self.model.named_parameters():
                    idxs_key = n + 'idxs'
                    vals_key = n + 'vals'
                    idxs = state_dict.get(idxs_key, None)
                    vals = state_dict.get(vals_key, None)

                    if idxs is not None and vals is not None:
                        idxs = idxs.to(self.config.device)
                        vals = vals.to(self.config.device)
                        
                        grad = self.transformer.decode(
                            self.compressor.decompress(
                                p.to(self.config.device),
                                idxs,
                                vals,
                                self.xshapes[n],
                                self.totalks[n],
                            )
                        ).to(self.config.device)

                        p.data.sub_(grad.sign(), alpha = self.scheduler.get_last_lr()[0] )

                # 10. Compute loss after gradient application        
                loss_after_own = 0.0
                n_batches = 0
                with torch.no_grad():
                    self.model.eval()
                    # Reuse same batches and indices for consistent comparison
                    for i, batch in enumerate(batches):
                        if i not in sampled_indices:
                            continue
                            
                        input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                        labels = input_ids.clone()
                        labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        loss_after_own += outputs.loss.item()
                        n_batches += 1
                        del input_ids, labels, outputs
                        torch.cuda.empty_cache()
                
                # Clean up stored batches
                del batches
                torch.cuda.empty_cache()


                loss_after_per_batch = loss_after_own / n_batches if n_batches > 0 else 0
                tplr.logger.info(f'Loss after (own data): {loss_after_per_batch}')

                # 11. Calculate improvements and update scores
                loss_improvement = loss_before_per_batch - loss_after_per_batch
                tplr.logger.info(f'Loss improvement (own data): {loss_improvement}')

                for n, p in self.model.named_parameters():
                    p.data.copy_(original_params[n])

                own_relative_improvement = loss_improvement / loss_before_per_batch if loss_before_per_batch > 0 else 0.0
                tplr.logger.info(f"Relative improvement (own data): {own_relative_improvement:.4f}")

                # 7. Load evaluation data from random page
                data_start = tplr.T()
                pages_random = await tplr.r2_dataset.R2DatasetLoader.next_pages(
                    offset=self.sync_window,
                    n_pages=self.hparams.pages_per_window,
                    seed=random.randint(0, 10000)
                )
                loader_random = await tplr.r2_dataset.R2DatasetLoader.create(
                    batch_size=self.hparams.batch_size,
                    sequence_length=self.hparams.sequence_length,
                    pages_info=pages_random,
                    tokenizer=self.tokenizer
                )
                tplr.logger.info(f'{tplr.P(self.sync_window, tplr.T() - data_start)} Loaded random evaluation  data')
                state_dict, _ = eval_result

                # 8. Compute initial loss
                self.optimizer.zero_grad()
                self.model.zero_grad()
                loss_before_random = 0.0
                n_batches = 0

                with torch.no_grad():
                    self.model.eval()
                    # First pass to count batches and store them
                    batches = []
                    for batch in loader_random:
                        batches.append(batch)
                    
                    total_batches = len(batches)
                    sample_size = max(1, int(total_batches * self.hparams.validator_sample_rate))
                    sampled_indices = random.sample(range(total_batches), sample_size)
                    sampled_indices = sorted(sampled_indices)  # Sort for sequential access
                    
                    tplr.logger.info(f"Evaluating {sample_size}/{total_batches} batches ({self.hparams.validator_sample_rate*100:.1f}%)")
                    
                    for i, batch in enumerate(batches):
                        if i not in sampled_indices:
                            continue
                            
                        input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                        labels = input_ids.clone()
                        labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        loss_before_random += outputs.loss.item()
                        n_batches += 1
                        del input_ids, labels, outputs
                        torch.cuda.empty_cache()

                loss_random_before_per_batch = loss_before_random / n_batches if n_batches > 0 else 0
                tplr.logger.info(f'Loss before (random data): {loss_random_before_per_batch}')

                # 9. Apply gradient and compute loss after
                self.optimizer.zero_grad()
                self.model.zero_grad()

                for n, p in self.model.named_parameters():
                    idxs_key = n + 'idxs'
                    vals_key = n + 'vals'
                    idxs = state_dict.get(idxs_key, None)
                    vals = state_dict.get(vals_key, None)

                    if idxs is not None and vals is not None:
                        idxs = idxs.to(self.config.device)
                        vals = vals.to(self.config.device)
                        
                        grad = self.transformer.decode(
                            self.compressor.decompress(
                                p.to(self.config.device),
                                idxs,
                                vals,
                                self.xshapes[n],
                                self.totalks[n],
                            )
                        ).to(self.config.device)

                        p.data.sub_(grad.sign(), alpha = self.scheduler.get_last_lr()[0] )

                # 10. Compute loss after gradient application        
                loss_after_random = 0.0
                n_batches = 0
                with torch.no_grad():
                    self.model.eval()
                    # Reuse same batches and indices for consistent comparison
                    for i, batch in enumerate(batches):
                        if i not in sampled_indices:
                            continue
                            
                        input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                        labels = input_ids.clone()
                        labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        loss_after_random += outputs.loss.item()
                        n_batches += 1
                        del input_ids, labels, outputs
                        torch.cuda.empty_cache()
                
                # Clean up stored batches
                del batches
                torch.cuda.empty_cache()

                loss_after_per_batch = loss_after_random / n_batches if n_batches > 0 else 0
                tplr.logger.info(f'Loss after (random data): {loss_after_per_batch}')

                # 11. Calculate improvements and update scores
                loss_improvement = loss_before_per_batch - loss_after_per_batch
                tplr.logger.info(f'Loss improvement (random data): {loss_improvement}')

                # Restore original model state
                for n, p in self.model.named_parameters():
                    p.data.copy_(original_params[n])

                # Cleanup
                del loader, loader_random, pages, pages_random
                torch.cuda.empty_cache()

                #  Restore Model to before eval state
                for n, p in self.model.named_parameters():
                    p.data.copy_(original_params[n])

                relative_improvement = loss_improvement / loss_before_per_batch if loss_before_per_batch > 0 else 0.0
                tplr.logger.info(f"Relative improvement (random data): {relative_improvement:.4f}")

                # Calculate original performance score (gradient quality)
                score = (loss_before_own - loss_after_own) / loss_before_own if loss_before_own > 0 else 0

                # Calculate binary indicator for overfitting detection
                improvement_own = (loss_before_own - loss_after_own) / loss_before_own if loss_before_own > 0 else 0
                improvement_random = (loss_before_random - loss_after_random) / loss_before_random if loss_before_random > 0 else 0
                binary_indicator = 1 if improvement_own > improvement_random else -1
                
                # Update binary indicator moving average (separate from score)
                if eval_uid not in self.binary_moving_averages:
                    self.binary_moving_averages[eval_uid] = 0.0
                
                self.binary_moving_averages[eval_uid] = (1 - self.hparams.ma_alpha) * self.binary_moving_averages[eval_uid] + self.hparams.ma_alpha * binary_indicator
                
                # Normalize binary moving average to [0,1] range
                normalized_binary_avg = (self.binary_moving_averages[eval_uid]) / 2
                
                # Calculate final score incorporating both metrics
                final_score = score * normalized_binary_avg
            
                self.evaluated_uids.add(eval_uid)

                torch.cuda.empty_cache()

                self.scores[eval_uid] = final_score
                # Ensure moving average score is non-negative
                self.moving_avg_scores[eval_uid] = max(self.hparams.ma_alpha * self.moving_avg_scores[eval_uid] + (1 - self.hparams.ma_alpha) * final_score, 0.0)

                # 12. Calculate weights using temperature-based softmax
                weights = torch.zeros_like(self.moving_avg_scores)
                evaluated_mask = torch.zeros_like(self.moving_avg_scores, dtype=torch.bool)
                evaluated_mask[list(self.evaluated_uids)] = True

                positive_mask = (self.moving_avg_scores > 0) & evaluated_mask
                
                if positive_mask.any():
                    # Apply normalization to all positive scores at once
                    weights[positive_mask] = min_power_normalization(
                        self.moving_avg_scores[positive_mask], 
                        power=self.hparams.power_normalisation
                    )
                    
                    # Log warning if weights don't sum to 1
                    weight_sum = weights.sum().item()
                    tplr.logger.debug(f"Weight sum: {weight_sum}")
                    if abs(weight_sum - 1.0) > 1e-6:
                        tplr.logger.warning(f"Weights sum to {weight_sum}, expected close to 1.0")
                else:
                    tplr.logger.info("No positive scores found, all weights set to 0")

                # 13. Log scores and metrics
                tplr.logger.info('Updated scores for evaluated UIDs:')
                for uid in self.evaluated_uids:
                    tplr.logger.info(f'UID {uid}:')
                    tplr.logger.info(f'  - Last score: {self.scores[uid]}')
                    tplr.logger.info(f'  - Moving avg score: {self.moving_avg_scores[uid]:.4f}')
                    tplr.logger.info(f'  - Weight: {weights[uid]:.4f}')
                    tplr.logger.info(f'  - Last Binary score: {self.binary[uid]:.4f}')
                    tplr.logger.info(f'  - Binary moving avg: {self.binary_moving_averages[uid]:.4f}') 
                    tplr.logger.info(f'  - Normalised Binary moving avg: {self.normalized_binary_avg[uid]:.4f}')

                # 14. Log wandb metrics
                valid_score_indices = torch.nonzero(self.scores > 0).squeeze().view(-1)
                for uid_i in valid_score_indices:
                    uid = uid_i.item()
                    self.wandb.log({
                        f"validator/scores/{uid}": self.scores[uid_i].item(),
                        f"validator/moving_avg_scores/{uid}": self.moving_avg_scores[uid_i].item(),
                        f"validator/weights/{uid}": weights[uid_i].item(),
                        f"validator/binary_scores/{uid}": binary_indicator,
                        f"validator/binary_moving_avg/{uid}": self.binary_moving_averages[uid],
                        f"validator/normalised_binary_moving_avg/{uid}": self.normalized_binary_avg[uid],
                    }, step=self.global_step)
                self.wandb.log({
                    "validator/loss/own/before": loss_before_own,
                    "validator/loss/own/after": loss_after_own,
                    "validator/loss/random/before": loss_before_random,
                    "validator/loss/random/after": loss_after_random,
                    "validator/loss/improvement": score,
                    "validator/network/block": self.current_block,
                    "validator/network/window": self.sync_window,
                    "validator/network/step": self.global_step,
                    "validator/network/evaluated_uids": len(self.evaluated_uids),
                    "validator/optimizer/learning_rate": self.scheduler.get_last_lr()[0],
                    "validator/network/active_miners": len(valid_score_indices),
                }, step=self.global_step)
                tplr.logger.info(f'{tplr.P(self.sync_window, tplr.T() - scoring_start)} Computed scores and weights')
            else:
                tplr.logger.info(f"No gradient received from UID {eval_uid}. Slashing moving average score by 50%.")
                # Reduce the moving average score by 50%
                old_score = self.moving_avg_scores[eval_uid].item()  # Get the actual value
                self.moving_avg_scores[eval_uid] *= 0.5  # Apply 50% reduction
                new_score = self.moving_avg_scores[eval_uid].item()  # Get new value for logging
                tplr.logger.info(f"Reduced moving average score of UID {eval_uid} from {old_score:.4f} to {new_score:.4f} due to missing gradient.")

                # Ensure the UID is included in evaluated_uids
                self.evaluated_uids.add(eval_uid)

                # Recalculate weights
                weights = torch.zeros_like(self.moving_avg_scores)
                evaluated_mask = torch.zeros_like(self.moving_avg_scores, dtype=torch.bool)
                evaluated_mask[list(self.evaluated_uids)] = True

                positive_mask = (self.moving_avg_scores > 0) & evaluated_mask

                if positive_mask.any():
                    # Apply normalization to all positive scores at once
                    weights[positive_mask] = min_power_normalization(
                        self.moving_avg_scores[positive_mask], 
                        power=self.hparams.power_normalisation
                    )
                    
                    # Log warning if weights don't sum to 1
                    weight_sum = weights.sum().item()
                    tplr.logger.debug(f"Weight sum: {weight_sum}")
                    if abs(weight_sum - 1.0) > 1e-6:
                        tplr.logger.warning(f"Weights sum to {weight_sum}, expected close to 1.0")
                else:
                    tplr.logger.info("No positive scores found, all weights set to 0")

                # Log updated scores
                tplr.logger.info('Updated scores for evaluated UIDs after slashing:')
                for uid in self.evaluated_uids:
                    tplr.logger.info(f'UID {uid}:')
                    tplr.logger.info(f'  - Moving avg score: {self.moving_avg_scores[uid]:.4f}')

                # Optionally, log to WandB
                self.wandb.log({
                    f"validator/moving_avg_scores/{eval_uid}": self.moving_avg_scores[eval_uid].item(),
                    f"validator/weights/{eval_uid}": weights[eval_uid].item(),
                }, step=self.global_step)
                tplr.logger.info(f'{tplr.P(self.sync_window, tplr.T() - scoring_start)} Computed scores and weights')
            
            tplr.logger.info(f'{tplr.P(self.sync_window, tplr.T() - eval_start)} Completed evaluation')
            # 15. Create checkpoints periodically
            if self.global_step % self.hparams.checkpoint_frequency == 0 and self.global_step != 0: 
                tplr.logger.info(f"Creating checkpoint at global_step {self.global_step}")

                checkpoint_data = {
                    'model_state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                    'optimizer_state_dict': {k: v.cpu().clone() if torch.is_tensor(v) else v 
                                           for k, v in self.optimizer.state_dict().items()},
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'momentum': {k: v.cpu().clone() for k, v in self.momentum.items()},
                    'start_window': self.start_window,
                    'current_window': self.current_window,
                }

                asyncio.create_task(
                    self.comms.put(
                        state_dict=checkpoint_data,
                        uid=str(self.uid),
                        window=self.current_window,
                        key='checkpoint',
                        global_step=self.global_step,
                        local=False
                    )
                )

            # 16. Update model with gathered gradients
            self.model.train()
            update_start = tplr.T()
            self.optimizer.zero_grad()
            self.model.zero_grad()
            # TODO: consider slashing here too
            if gather_result is not None and gather_result.state_dict is not None:
                for n, p in self.model.named_parameters():
                    idxs_key = n + 'idxs'
                    vals_key = n + 'vals'
                    idxs = getattr(gather_result.state_dict, idxs_key, None)
                    vals = getattr(gather_result.state_dict, vals_key, None)
                    if idxs is not None and vals is not None:
                        if not isinstance(idxs, (list, tuple)):
                            idxs = [idxs]
                        if not isinstance(vals, (list, tuple)):
                            vals = [vals]
                        
                        new_grad = self.transformer.decode(
                            self.compressor.batch_decompress(
                                p.to(self.config.device),
                                idxs,
                                vals,
                                self.xshapes[n],
                                self.totalks[n],
                            )
                        )
                        # Store pre-sign gradient in momentum
                        self.momentum[n] = new_grad.clone()
                        
                        p.data.sub_(grad.sign(), alpha = self.scheduler.get_last_lr()[0] )
                    else:
                        tplr.logger.info(f"Gradient data missing for parameter {n}, skipping.")
            tplr.logger.info(f'{tplr.P(self.sync_window, tplr.T() - update_start)} Updated model')

            self.optimizer.step()
            self.scheduler.step()
            torch.cuda.empty_cache()

            # 17. Set weights periodically
            if self.sync_window % self.hparams.windows_per_weights == 0:
                self.subtensor.set_weights(
                    wallet=self.wallet,
                    netuid=self.config.netuid,
                    uids=self.metagraph.uids,
                    weights=weights,
                    wait_for_inclusion=False,
                    wait_for_finalization=False,
                )
            # Log total window time and metrics
            tplr.logger.info(f'{tplr.P(self.sync_window, tplr.T() - window_start)} Completed window iteration')
            
            self.wandb.log({
                "validator/timing/window_total": tplr.T() - window_start,
                "validator/timing/peer_update": tplr.T() - peer_start,
                "validator/timing/gather": tplr.T() - gather_start,
                "validator/timing/evaluation": tplr.T() - eval_start,
                "validator/timing/model_update": tplr.T() - update_start,
            }, step=self.global_step)
            
            # 18. Increment global step
            self.global_step += 1

    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event['header']['number']) #type : ignore
            new_window = int(self.current_block / self.hparams.blocks_per_window)
            if new_window != self.current_window:
                self.current_window = new_window
                self.comms.current_window = self.current_window  # Synchronize comms current_window
        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler)
                break
            except Exception:
                time.sleep(1)

def min_power_normalization(logits, power=2.0, epsilon=1e-8):
    """Normalizes logits using a minimum power normalization approach.
    
    This function applies power normalization to the input logits, raising them to a power
    and normalizing to create a probability distribution. If the sum is too small (below epsilon),
    returns zeros to avoid division by very small numbers.

    Args:
        logits (torch.Tensor): Input tensor to be normalized
        power (float, optional): Power to raise the logits to. Defaults to 2.0.
        epsilon (float, optional): Small value to prevent division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: Normalized probabilities
    """
    if logits.dim() == 0:
        logits = logits.unsqueeze(0)
    
    powered_logits = logits ** power
    sum_powered = torch.sum(powered_logits)
    if sum_powered > epsilon:
        probabilities = powered_logits / sum_powered
    else:
        probabilities = torch.zeros_like(powered_logits)
    
    return probabilities


if __name__ == "__main__":
    asyncio.run(Validator().run())
