# The MIT License (MIT)
# © 2025 tplr.ai

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

        # Init score tracking variables
        self.loss_before_per_batch_own = 0.0
        self.loss_after_per_batch_own = 0.0
        self.loss_before_per_batch_random = 0.0
        self.loss_after_per_batch_random = 0.0
        self.loss_improvement_own = 0.0
        self.loss_improvement_random = 0.0
        self.relative_improvement_own = 0.0
        self.relative_improvement_random = 0.0
        self.valid_score_indices = []
        self.gradient_scores = torch.zeros(self.metagraph.n, dtype=torch.float32)
        self.binary_indicator_scores = torch.zeros(self.metagraph.n, dtype=torch.float32)
        self.gradient_moving_avg_scores = torch.zeros(self.metagraph.n, dtype=torch.float32) 
        self.final_moving_avg_scores = torch.zeros(self.metagraph.n, dtype=torch.float32)
        self.binary_moving_averages = torch.zeros(self.metagraph.n, dtype=torch.float32)
        self.weights = torch.zeros(self.metagraph.n, dtype=torch.float32)
        self.normalised_binary_moving_averages = torch.zeros(self.metagraph.n, dtype=torch.float32)
        self.evaluated_uids = set()  

        # Add step tracking
        self.window_step = 0
        self.eval_count = 0  
        
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


    async def run(self):
        # Start background block listener
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(
            target=self.block_listener, 
            args=(self.loop,), 
            daemon=True
        ).start()
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
        tplr.logger.info("Loaded commitments")

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
        # self.loop = asyncio.get_running_loop()

        self.comms.start_commitment_fetcher()
        self.comms.start_background_tasks()

        while True:
            # Check for catch-up need
            catch_up_success, new_global_step, new_optimizer, new_scheduler = await self.comms.check_and_perform_catch_up(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                transformer=self.transformer,
                compressor=self.compressor,
                current_window=self.current_window,
                sync_window=self.sync_window,
                device=self.config.device,
                peers=self.peers,
                uid=self.uid,
                global_step=self.global_step,
                hparams=self.hparams
            )
            
            if catch_up_success:
                self.global_step = new_global_step
                self.optimizer = new_optimizer
                self.scheduler = new_scheduler
                self.sync_window = self.current_window
                continue

            # Wait for offset
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
            
            newly_inactive = self.comms.inactive_peers
            current_window = self.sync_window
            
            # Process newly inactive peers
            for uid in newly_inactive:
                if uid not in self.inactive_scores:
                    self.inactive_scores[uid] = (current_window, self.final_moving_avg_scores[uid].item())
                    tplr.logger.info(f"UID {uid} became inactive at window {current_window} with score {self.final_moving_avg_scores[uid].item():.4f}")
            
            # Apply penalties to all inactive peers
            for uid, (_, _) in list(self.inactive_scores.items()):
                # If peer became active again, remove from inactive tracking
                if uid in self.eval_peers:
                    del self.inactive_scores[uid]
                    tplr.logger.info(f"UID {uid} became active again")
                    continue
                
                # Apply flat 25% penalty instead of exponential decay
                old_score = self.final_moving_avg_scores[uid].item()
                self.final_moving_avg_scores[uid] *= 0.75  # Apply flat 25% reduction
                new_score = self.final_moving_avg_scores[uid].item()
                
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
                timeout=30,
                device=self.config.device,
                local=False,
                stale_retention=100,
                global_step=self.global_step,
                store_gathers=self.config.store_gathers
            )
            tplr.logger.info(f'{tplr.P(self.sync_window, tplr.T() - gather_start)} Gathered gradients from peers')

            # Add check for empty peers (evaluating all peer uids)
            if not self.peers:
                tplr.logger.warning(f"No peers available for evaluation in window {self.sync_window}. Waiting for next window.")
                self.global_step += 1
                continue

            # 5. Save original model state for evaluation
            eval_start = tplr.T()
            # Sample a random subset of evaluation peers based on hparam uids_per_window
            evaluation_uids = random.sample(
                self.eval_peers,
                min(self.hparams.uids_per_window, len(self.eval_peers))
            )
            tplr.logger.info(f'Evaluating random subset of peers: {evaluation_uids}')
            for eval_uid in evaluation_uids:
                tplr.logger.info(f'Evaluating uid: {eval_uid}')

                eval_result = await self.comms.get(
                    uid=str(eval_uid),
                    window=self.sync_window,
                    key='gradient',
                    timeout=30,
                    local=False,
                    stale_retention=10
                )

                if eval_result is not None and eval_result[0] is not None:
                    state_dict, _ = eval_result
                    eval_payload = await tplr.evaluation.evaluate_peer(
                        uid=eval_uid,
                        state_dict=state_dict,
                        sync_window=self.sync_window,
                        hparams=self.hparams,
                        tokenizer=self.tokenizer,
                        config=self.config,
                        model=self.model,
                        transformer=self.transformer,
                        compressor=self.compressor,
                        xshapes=self.xshapes,
                        totalks=self.totalks,
                        device=self.config.device,
                        lr=self.scheduler.get_last_lr()[0],
                        optimizer=self.optimizer,
                        scheduler=self.scheduler
                    )
                    self.gradient_scores[eval_uid] = eval_payload["gradient_score"]
                    self.binary_indicator_scores[eval_uid] = eval_payload["binary_indicator"]
                    self.loss_before_per_batch_own = eval_payload["loss_before_per_batch_own"]
                    self.loss_after_per_batch_own  = eval_payload["loss_after_per_batch_own"]
                    self.relative_improvement_own    = eval_payload["relative_improvement_own"]
                    self.loss_before_per_batch_random = eval_payload["loss_before_per_batch_random"]
                    self.loss_after_per_batch_random  = eval_payload["loss_after_per_batch_random"]
                    self.relative_improvement_random    = eval_payload["relative_improvement_random"]
                    self.evaluated_uids.add(eval_uid)
                else:
                    tplr.logger.info(f"No gradient received from UID {eval_uid}. Slashing moving average score by 50%.")
                    old_score = self.final_moving_avg_scores[eval_uid].item()
                    self.final_moving_avg_scores[eval_uid] *= 0.5
                    new_score = self.final_moving_avg_scores[eval_uid].item()
                    tplr.logger.info(f"Reduced moving average score of UID {eval_uid} from {old_score:.4f} to {new_score:.4f}")
                    self.evaluated_uids.add(eval_uid)

            # Log scores and metrics for evaluated UIDs
            tplr.logger.info('Updated scores for evaluated UIDs:')
            for uid in self.evaluated_uids:
                tplr.logger.info(f'UID {uid}:')
                tplr.logger.info(f'  - Last score: {self.gradient_scores[uid]}')
                tplr.logger.info(f'  - Binary indicator: {self.binary_indicator_scores[uid]:.4f}')
                tplr.logger.info(f'  - Binary moving avg: {self.binary_moving_averages[uid]:.4f}')
                tplr.logger.info(f'  - Normalised binary score: {self.normalised_binary_moving_averages[uid]:.4f}')
                tplr.logger.info(f'  - Final Moving avg score: {self.final_moving_avg_scores[uid]:.4f}')
                tplr.logger.info(f'  - Weight: {self.weights[uid]:.4f}')
            
            # Log WandB metrics per UID
            for uid in sorted(self.evaluated_uids):
                self.wandb.log({
                    f"validator/gradient_scores/{uid}": self.gradient_scores[uid].item(),
                    f"validator/binary_indicators/{uid}": self.binary_indicator_scores[uid].item(),
                    f"validator/binary_moving_averages/{uid}": self.binary_moving_averages[uid].item(),
                    f"validator/normalised_binary_scores/{uid}": self.normalised_binary_moving_averages[uid].item(),
                    f"validator/final_moving_avg_scores/{uid}": self.final_moving_avg_scores[uid].item(),
                    f"validator/weights/{uid}": self.weights[uid].item(),
                }, step=self.global_step)

            # 17. Set weights periodically
            if self.sync_window % self.hparams.windows_per_weights == 0:
                # Only set weights for evaluated peers with non-negative (positive) weight values.
                positive_weighted_uids = sorted([uid for uid in self.evaluated_uids if self.weights[uid] > 0])
                if positive_weighted_uids:
                    self.subtensor.set_weights(
                        wallet=self.wallet,
                        netuid=self.config.netuid,
                        uids=positive_weighted_uids,
                        weights=self.weights[positive_weighted_uids],
                        wait_for_inclusion=False,
                        wait_for_finalization=False,
                    )

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
            
            # 16. Merge the gathered gradients into the model AFTER finishing evaluation
            self.model.train()
            update_start = tplr.T()
            self.optimizer.zero_grad()
            self.model.zero_grad()
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
                        if p.grad is None:
                            p.grad = new_grad
                        else:
                            p.grad.copy_(new_grad)
                        p.grad.sign_()                        
                    else:
                        tplr.logger.info(f"Gradient data missing for parameter {n}, skipping.")
            tplr.logger.info(f'{tplr.P(self.sync_window, tplr.T() - update_start)} Updated model')
            
            self.optimizer.step()
            self.scheduler.step()
            torch.cuda.empty_cache()
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
