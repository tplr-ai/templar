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
import argparse
import asyncio
import bittensor as bt
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import random
import sys
import threading
import time
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM
import pandas as pd
import wandb
import wandb.plot

# Local imports.
import templar as tplr

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Validator:

    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Validator script')
        parser.add_argument('--project', type=str, default='templar', help='Optional wandb project name')
        parser.add_argument('--netuid', type=int, default=3, help='Bittensor network UID.')
        parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
        parser.add_argument('--actual_batch_size', type=int, default=8, help='Training batch size per accumulation.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
        parser.add_argument('--remote', action='store_true', help='Connect to other buckets')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        parser.add_argument('--sync_state', action='store_true', help='Syncs the model state by pulling from the history.')
        parser.add_argument('--test', action='store_true', help='Run on test network')
        parser.add_argument('--local', action='store_true', help='Run on local network')
        parser.add_argument('--no_autoupdate', action='store_true', help='Disable automatic updates')
        parser.add_argument("--process_name", type=str, help="The name of the PM2 process")
        parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to save/load the checkpoint. If None, the path is set to checkpoint-V<UID>.pth.')
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        config = bt.config(parser)
        if config.test:
            config.subtensor.network = 'test'
            config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'
        elif config.local:
            config.subtensor.network = 'local'
            config.subtensor.chain_endpoint = 'ws://127.0.0.1:9944'
        if config.debug: tplr.debug()
        if config.trace: tplr.trace()
        tplr.validate_bucket_or_exit(config.bucket)
        if not config.no_autoupdate:
            autoupdater = tplr.AutoUpdate(process_name=config.process_name, bucket_name=config.bucket)
            autoupdater.start()
        return config

    def __init__(self):
        # Init config.
        self.config = Validator.config()
        tplr.logger.info('\n' + '-' * 40 + ' Config ' + '-' * 40)
        tplr.logger.info(self.config)

        # Init bittensor objects.
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            tplr.logger.error(f'\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}[/bold]. You need to register first with: [blue]`btcli subnet register`[/blue]\n')
            sys.exit()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        tplr.logger.info('\n' + '-' * 40 + ' Objects ' + '-' * 40)
        tplr.logger.info(f'\nWallet: {self.wallet}\nSubtensor: {self.subtensor}\nMetagraph: {self.metagraph}\nUID: {self.uid}')

        # Init bucket.
        try:
            if self.config.bucket != self.subtensor.get_commitment(self.config.netuid, self.uid):
                raise ValueError('')
        except:
            self.subtensor.commit(self.wallet, self.config.netuid, self.config.bucket)
        tplr.logger.info('Bucket:' + self.config.bucket)

        # Init Wandb.
        # Ensure the wandb directory exists
        wandb_dir = os.path.join(os.getcwd(), 'wandb')
        os.makedirs(wandb_dir, exist_ok=True)

        # Define the run ID file path inside the wandb directory
        run_id_file = os.path.join(wandb_dir, f"wandb_run_id_V{self.uid}_{tplr.__version__}.txt")

        # Attempt to read the existing run ID
        if os.path.exists(run_id_file):
            with open(run_id_file, 'r') as f:
                run_id = f.read().strip()
            tplr.logger.info(f"Resuming WandB run with id {run_id}")
        else:
            run_id = None
            tplr.logger.info("Starting a new WandB run.")

        # Initialize WandB
        self.wandb = tplr.initialize_wandb(
            run_prefix='V',
            uid=self.uid,
            config=self.config,
            group='validator',
            job_type='validation'
        )
        self.metrics_history = pd.DataFrame()
        # Init model.
        tplr.logger.info('\n' + '-' * 40 + ' Hparams ' + '-' * 40)
        self.hparams = tplr.load_hparams()
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        self.model = LlamaForCausalLM(config=self.hparams.model_config)
        self.model.to(self.config.device)
        self.model.eval()

        # Set checkpoint path
        self.checkpoint_path = f"checkpoint-V{self.uid}.pth" if self.config.checkpoint_path is None else self.config.checkpoint_path 

        # Load checkpoint if it exists
        if os.path.exists(self.checkpoint_path):
            tplr.logger.info(f"Loading checkpoint from {self.checkpoint_path}")

            # The load_checkpoint function updates the model's state in place
            global_step, additional_state = asyncio.run(tplr.load_checkpoint(
                filename=self.checkpoint_path,
                model=self.model,            # Model state is updated inside load_checkpoint
                device=self.config.device
            ))

            # Update global_step
            self.global_step = global_step

            # Load additional state variables
            self.scores = additional_state.get('scores', torch.zeros(256, dtype=torch.float32))
            self.weights = additional_state.get('weights', torch.zeros(256, dtype=torch.float32))

            tplr.logger.info(f"Resumed from global step {self.global_step}")
        else:
            tplr.logger.info("No checkpoint found. Starting from scratch.")
            self.global_step = 0
            self.scores = torch.zeros(256, dtype=torch.float32)
            self.weights = torch.zeros(256, dtype=torch.float32)
        
        # Init buckets.
        self.buckets = []
        for uid in self.metagraph.uids:
            try:
                bucket =  self.subtensor.get_commitment(self.config.netuid, uid)
                tplr.logger.debug(f"Retrieved bucket for UID {uid}: {bucket}")
                self.buckets.append(bucket)
            except Exception as e:
                tplr.logger.debug(f"Failed to retrieve bucket for UID {uid}: {e}")
                self.buckets.append(None)


        self.last_window = 0
        self.optimal_pages_per_step = 4
        self.current_block = self.subtensor.block
        self.current_window = self.block_to_window( self.current_block )
        self.window_seeds = {self.current_window: self.window_to_seed( self.current_window) }
        self.block_event = asyncio.Event()
        self.new_window_event = asyncio.Event()
        self.stop_event = asyncio.Event()     
        self.step_scores = torch.zeros( 256, dtype = torch.float32 ) 
        self.scores = torch.zeros( 256, dtype = torch.float32 ) 
        self.weights = torch.zeros( 256, dtype = torch.float32 ) 
        self.sample_rate = 1.0
        print ( self.hparams )
        
    async def update(self):
        """Continuously updates the global state by polling every 10 minutes."""
        while not self.stop_event.is_set():
            st = tplr.T()
            await asyncio.to_thread(self.perform_update)
            tplr.logger.info(f"{tplr.P(self.current_window, tplr.T() - st)} Updated global state.")
            await asyncio.sleep(600)

    def perform_update(self):
        """Updates subtensor connection, metagraph, hyperparameters and buckets."""
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.hparams = tplr.load_hparams()

        next_buckets = []
        for uid in self.metagraph.uids:
            try:
                bucket = self.config.bucket if not self.config.remote else self.subtensor.get_commitment(self.config.netuid, uid)
                if tplr.is_valid_bucket(bucket):
                    tplr.logger.debug(f"UID {uid}: Valid bucket found: {bucket}")
                    next_buckets.append(bucket)
                else:
                    tplr.logger.debug(f"UID {uid}: Invalid or missing bucket name: {bucket}")
                    next_buckets.append(None)
            except Exception as e:
                tplr.logger.warning(f"UID {uid}: Error retrieving bucket: {e}")
                next_buckets.append(None)

        self.buckets = next_buckets

    async def run(self):
        # Main loop.
        self.loop = asyncio.get_running_loop()
        self.update_task = asyncio.create_task(self.update())
        self.listener = threading.Thread(target=self.block_listener, args=(self.loop,), daemon=True).start()
        
        # Optionally sync the model state by pulling model states from the history.
        if self.config.sync_state:
            st = tplr.T()
            history_windows = [ self.current_window - i for i in range (self.hparams.max_history) ]
            state_slices = await tplr.download_slices_for_buckets_and_windows(
                buckets = self.buckets,
                windows = history_windows,
                key = 'state'
            )
            for window in tqdm(history_windows, desc="Syncing state"):
                max_global_step = await tplr.apply_slices_to_model( 
                    model=self.model, 
                    window=window,
                    seed=window,
                    compression=self.hparams.compression,
                    key='state',
                )
                if max_global_step is not None:
                    self.global_step = max(self.global_step, max_global_step)
                tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Applied historical state and updated global step to {self.global_step}.")
            torch.cuda.empty_cache()

        # Run validation.
        while True:
            try:
                # Get the window we are evalling.
                tplr.logger.info('[bold]' + '\n' + '-' * 40 + f' Step: {self.global_step} ' + '-' * 40)
                gs_start = tplr.T()
                self.global_step += 1
                offset = 2
                window = self.current_window - offset


                # Save checkpoint every 500 steps
                if self.global_step % 500 == 0:
                    tplr.logger.info(f"Scheduling checkpoint save at block {self.global_step}")
                    # Update last checkpoint block to avoid repeated saves at the same block
                    self.last_checkpoint_block = self.global_step
                    # Schedule the save_checkpoint function to run asynchronously
                    asyncio.create_task(tplr.save_checkpoint(
                        filename=self.checkpoint_path,
                        model=self.model,
                        global_step=self.global_step,
                        scores=self.scores,
                        weights=self.weights
                    ))
                
                # Download the state for the eval window.
                st = tplr.T()
                state_slices = await tplr.download_slices_for_buckets_and_windows(
                    buckets = self.buckets,
                    windows = [ window ],
                    key = 'state'
                )
                n_state_slices = len(state_slices[ window ]) if window in state_slices else 0
                tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Downloaded {n_state_slices} window states.")
                
                # Download the delta for the eval window.
                st = tplr.T()
                eval_slices = await tplr.download_slices_for_buckets_and_windows(
                    buckets = self.buckets,
                    windows = [ window ],
                    key = 'delta'
                ) 
                n_eval_slices = len(eval_slices[ window ]) if window in eval_slices else 0
                tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Downloaded {n_eval_slices} window deltas.")                
                if n_eval_slices == 0:
                    tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: No slices to eval, continue ...")
                    while self.current_window - offset == window: await asyncio.sleep(0.1) # Wait for next window.
                    continue
                
                # Applied the model  state for the eval window.
                st = tplr.T()
                max_global_step = await tplr.apply_slices_to_model( 
                    model=self.model, 
                    window=window,
                    seed=window,
                    compression=self.hparams.compression,
                    key='state',
                )
                if max_global_step is not None:
                    self.global_step = max(self.global_step, max_global_step)
                tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Applied window state and updated global step to {self.global_step}.")
                
                # Obtain the indicies for the eval window.
                st = tplr.T()
                indices = await tplr.get_indices_for_window(
                    model = self.model,
                    seed = window,
                    compression = self.hparams.compression
                ) 
                tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Obtained window indices.")
                               

                # Attain the UID of this slice.
                st = tplr.T()
                valid_eval_slices = [s for s in eval_slices[window] if getattr(s, 'version', None) == tplr.__version__]
                if not valid_eval_slices:
                    tplr.logger.warning(f"{tplr.P(window, tplr.T() - st)}: No valid slices with matching version {tplr.__version__}, continuing...")
                    while self.current_window - offset == window:
                        await asyncio.sleep(0.1)  # Wait for next window.
                    continue
                eval_slice_info = random.choice( valid_eval_slices)                
                try: eval_uid = self.metagraph.hotkeys.index(eval_slice_info.hotkey)
                except ValueError:
                    tplr.logger.warning(f"{tplr.P(window, tplr.T() - st)}: {eval_slice_info.hotkey} not found in metagraph")
                    continue                                
                eval_slice_data = await tplr.get_slices( eval_slice_info.temp_file, self.model.device )                
                tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Loaded window slices for uid: [dark_sea_green]{eval_uid}[/dark_sea_green].")

                # Download the eval page for this uid.
                st = tplr.T()
                eval_pages = await tplr.dataset.DatasetLoader.next_pages(
                    offset = window,
                    n_pages = self.hparams.validator_window_eval_size,
                    seed = eval_uid
                )            
                random.shuffle( eval_pages )    
                eval_dataset = await tplr.dataset.DatasetLoader.create(
                    batch_size = self.config.actual_batch_size,
                    sequence_length = self.hparams.sequence_length,
                    pages_info = eval_pages,
                    tokenizer = self.hparams.tokenizer
                )                
                tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Downloaded eval pages: [light_steel_blue]{[p[1] for p in eval_pages]}[/light_steel_blue].")
  
                # Accumulate gradients from this page.
                eval_start = tplr.T()
                self.model.zero_grad()
                total_loss = 0.0
                full_steps = 0; total_steps = 0; 
                exhuasted_window = False
                with torch.enable_grad():
                    for idx, batch in enumerate(eval_dataset):
                        total_steps += 1
                        if random.random() < self.sample_rate and not exhuasted_window:
                            full_steps += 1
                            input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                            labels = input_ids.clone()
                            labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)
                            with torch.amp.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):  # Enable autocasting
                                outputs = self.model(input_ids=input_ids, labels=labels)
                            total_loss += outputs.loss.item()
                            outputs.loss.backward()
                            if self.current_window - offset != window: exhuasted_window = True; continue
                step_loss = total_loss/(full_steps+1)
                eval_duration = tplr.T() - eval_start
                tokens_per_step = self.hparams.sequence_length * self.config.actual_batch_size * (full_steps + 1)

                tokens_per_second = tokens_per_step / eval_duration
                tplr.logger.info(f"{tplr.P(window, eval_duration)}: Accumulated gradients:")
                tplr.logger.info(f"{tplr.P(window, eval_duration)}: \tTotal steps: [tan]{full_steps}/{total_steps}[/tan], Rate: [tan]{(full_steps/total_steps):.2f}[/tan], Target: [tan]{self.sample_rate:.2f}[/tan]")
                tplr.logger.info(f"{tplr.P(window, eval_duration)}: \tTotal tokens: [tan]{tokens_per_step}[/tan], Tokens per second: [tan]{tokens_per_second:.2f}[/tan]")
                tplr.logger.info(f"{tplr.P(window, eval_duration)}: \tLoss: [tan]{step_loss}[tan]")
                if exhuasted_window: self.sample_rate = max(0.0001, self.sample_rate * 0.95)
                else: self.sample_rate = min(1, self.sample_rate * 1.05)
                
                # Compute the score for this slice.
                st = tplr.T()
                score = 0.0 
                for i, (name_i, param_i) in enumerate( self.model.named_parameters() ):
                    if param_i.grad is None: continue  # Skip parameters without gradients
                    idxs_i = indices[name_i].to(self.model.device)
                    grad_i = param_i.grad.view(-1).clone()[idxs_i].to(self.model.device) 
                    slice_i = eval_slice_data[name_i].view(-1).to(self.model.device) 
                    theta_i = param_i.data.view(-1)[idxs_i]
                    delta_i = theta_i - slice_i
                    sim_i = torch.nn.functional.cosine_similarity(delta_i, grad_i, dim=0).item()
                    weight_i = param_i.data.view(-1)[idxs_i].norm().item() + 1e-8
                    score += weight_i * sim_i
                tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Computed score: [bold dark_sea_green]{score:.4f}[/bold dark_sea_green]")           

                # Assign and log scores.
                start_time = tplr.T()
                self.step_scores[ eval_uid ] = score
                self.scores[ eval_uid ] = (1 - self.hparams.validator_moving_alpha) * score + self.hparams.validator_moving_alpha * self.scores[eval_uid]
                # Store the original weights before adjustment.
                total_score = torch.sum(self.scores).item()
                if total_score == 0.0:
                    tplr.logger.warning("Total score is zero; setting all weights to zero.")
                    self.weights = torch.zeros_like(self.scores)
                    original_weights = self.weights.clone()  # All zeros in this case
                else:
                    # Normalize scores to get initial weights
                    self.weights = self.scores / total_score

                    # Store the original weights before adjustment
                    original_weights = self.weights.clone()

                    # Set negative weights to zero
                    negative_weights = self.weights < 0
                    if negative_weights.any():
                        tplr.logger.info("Negative weights detected; setting them to zero.")
                        self.weights[negative_weights] = 0.0

                        # Re-normalize the positive weights to ensure the sum is 1
                        positive_total = torch.sum(self.weights).item()
                        if positive_total == 0.0:
                            tplr.logger.warning("All weights are zero after removing negative weights; cannot normalize.")
                            # Weights remain zero
                        else:
                            self.weights = self.weights / positive_total

                # Log updated scores and weights
                valid_score_indices = torch.nonzero(self.scores != 0).squeeze().view(-1)
                for uid_i in valid_score_indices:
                    uid = uid_i.item()
                    moving_score = self.scores[uid].item()
                    weight = self.weights[uid].item()
                    orig_weight = original_weights[uid].item()
                    step_score = self.step_scores[uid].item()
                    tplr.logger.info(
                        f"\tuid: [dark_sea_green]{uid}[/dark_sea_green], "
                        f"step_score: [dark_sea_green]{step_score:.3f}[/dark_sea_green], "
                        f"moving_score: [dark_sea_green]{moving_score:.3f}[/dark_sea_green], "
                        f"weight: [dark_sea_green]{weight:.3f}[/dark_sea_green], "
                        f"original_weight: [dark_sea_green]{orig_weight:.3f}[/dark_sea_green]"
                    )
                
                # Apply all deltas to the model state.
                st = tplr.T()
                max_global_step = await tplr.apply_slices_to_model( 
                    model=self.model, 
                    window=window,
                    seed=window,
                    compression=self.hparams.compression,
                    key='delta',
                )
                if max_global_step is not None:
                    self.global_step = max(self.global_step, max_global_step)
                tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Applied window delta and updated global step to {self.global_step}.")
                
                # Clean local and remote space from old slices.
                st = tplr.T()
                await tplr.delete_files_before_window( window_max = window - self.hparams.max_history, key = 'state')
                await tplr.delete_files_before_window( window_max = window - self.hparams.max_history, key = 'delta')
                await tplr.delete_files_from_bucket_before_window( bucket = self.config.bucket, window_max = window - self.hparams.max_history, key = 'state' )
                await tplr.delete_files_from_bucket_before_window( bucket = self.config.bucket, window_max = window - self.hparams.max_history, key = 'delta' )
                tplr.logger.info(f"{tplr.P(window, tplr.T() - st)}: Cleaned file history.")

                # Finish step.
                gs_end = tplr.T()
                while self.current_window - offset == window:
                    await asyncio.sleep(0.1)
                window_time_delta = self.window_time - gs_end
                window_delta_str = f"[red]{window_time_delta:.2f}[/red]" if window_time_delta < 0 else f"[green]+{window_time_delta:.2f}[/green]"
                tplr.logger.info(f"{tplr.P(window, gs_end - gs_start)}[{window_delta_str}]: Finished step.")
                # Log main metrics
                wandb.log({
                    "validator/loss": step_loss,
                    "validator/tokens_per_step": tokens_per_step,
                    "validator/tokens_per_second": tokens_per_second,
                    "validator/sample_rate": self.sample_rate,
                    "validator/utilization": eval_duration / (gs_end - gs_start)
                }, step=self.global_step)                   

                # Prepare a list to hold metrics for all UIDs
                metrics_list = []

                # Collect metrics for all UIDs
                for uid_i in valid_score_indices:
                    uid = uid_i.item()
                    uid_str = str(uid)

                    # Extract metrics
                    step_score = self.step_scores[uid].item()
                    moving_score = self.scores[uid].item()
                    weight = self.weights[uid].item()
                    orig_weight = original_weights[uid].item()

                    # Append to metrics list for aggregation
                    metrics_list.append({
                        'global_step': self.global_step,
                        'uid': uid_str,
                        'step_score': step_score,
                        'moving_score': moving_score,
                        'weight': weight,
                        'original_weight': orig_weight,
                    })

                # Convert metrics list to DataFrame
                metrics_df = pd.DataFrame(metrics_list)

                # Append to metrics history
                self.metrics_history = pd.concat([self.metrics_history, metrics_df], ignore_index=True)

                # Drop duplicates to avoid multiple entries for the same step and UID
                self.metrics_history.drop_duplicates(subset=['global_step', 'uid'], keep='last', inplace=True)

                # List of metrics to plot
                metrics_to_plot = ['step_score', 'moving_score', 'weight', 'original_weight']

                # Create aggregated plots for each metric
                for metric_name in metrics_to_plot:
                    # Pivot the DataFrame to have UIDs as columns
                    pivot_df = self.metrics_history.pivot(index='global_step', columns='uid', values=metric_name).reset_index()

                    # Create wandb.Table
                    table = wandb.Table(dataframe=pivot_df)

                    # Get the list of UIDs (column names excluding 'global_step')
                    uids = pivot_df.columns.drop('global_step').tolist()

                    # Prepare xs (global steps) and ys (metric values per UID)
                    xs = pivot_df['global_step'].values.tolist()
                    ys = [pivot_df[uid].values.tolist() for uid in uids]

                    # Create the line plot
                    line_plot = wandb.plot.line_series(
                        xs=xs,
                        ys=ys,
                        keys=uids,
                        title=f"Validator {metric_name.replace('_', ' ').title()}",
                        xname="Global Step"
                    )
                    # Log the plot
                    
                    wandb.log({f"validator/{metric_name}": line_plot}, step=self.global_step)
                # Set temperatured weights on the chain.
                if self.current_block % 100 == 0:
                    tplr.logger.info(f"Setting weights on chain: {self.weights[ self.metagraph.uids ]}")
                    try:
                        result = self.subtensor.set_weights(
                            wallet = self.wallet,
                            netuid = self.config.netuid,
                            uids = self.metagraph.uids,
                            weights = self.weights[ self.metagraph.uids ],
                            wait_for_inclusion = True,
                            wait_for_finalization = False,
                        )
                        tplr.logger.info(f"Successfully set weights on chain: {result}")
                    except Exception as e:
                        tplr.logger.error(f"Failed to set weights on chain: {e}")
                                                                
            # Catch keyboard interrrupt.
            except KeyboardInterrupt:
                tplr.logger.info("Training interrupted by user. Stopping the run.")
                self.stop_event.setplr.T()
                await self.update_task
                sys.exitplr.T(0)
            
            # Catch unknown.
            except Exception as e:
                tplr.logger.exception(f"Exception during training loop: {e}")
                continue

    # Returns the slice window based on a blotplr.
    def block_to_window(self, block: int) -> int:
        return int(block / self.hparams.window_length)
    
    # Returns the slice window based on a blotplr.
    def window_to_seed(self, window: int) -> int:
        return str( self.subtensor.get_block_hash( window * self.hparams.window_length ) )

    # A listener thread which posts the block event
    # when the chain announces a new blotplr.
    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event['header']['number'])
            loop.call_soon_threadsafe(self.block_event.set)
            if self.block_to_window(self.current_block) != self.current_window:
                self.window_seeds[ self.block_to_window(self.current_block) ] = self.window_to_seed( self.block_to_window(self.current_block) )
                self.current_window = self.block_to_window(self.current_block)
                self.window_duration = tplr.T() - self.window_time if hasattr(self, 'window_time') else 0
                self.window_time = tplr.T()
                loop.call_soon_threadsafe(self.new_window_event.set)
                tplr.logger.info(f"{tplr.P(self.current_window, self.window_duration)} New Window.")
                
        # Run listener with retry.
        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler); break
            except Exception as e:
                 # Wait for 5 seconds before retrying
                tplr.logger.error(f"Failed to subscribe to block headers: {e}.\nRetrying in 1 seconds...")
                time.sleep(1) 
            
if __name__ == "__main__":
    validator = Validator()
    asyncio.run(validator.run())
