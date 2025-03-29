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

import asyncio
import argparse
import threading
import time
from typing import cast
from bittensor.core.subtensor import ScaleObj
import bittensor as bt
from datetime import datetime, timezone, timedelta
import gc

# Import tplr functions
import tplr
from transformers import LlamaConfig, LlamaForCausalLM


class AggregationServer:
    @staticmethod
    def agg_config():
        parser = argparse.ArgumentParser(description="Aggregation Server")
        parser.add_argument(
            "--netuid", type=int, default=3, help="Bittensor network UID."
        )
        parser.add_argument(
            "--wait-time",
            type=float,
            default=0.1,
            help="Additional wait time after window ends",
        )
        parser.add_argument(
            "--device", type=str, default="cuda", help="Device to use for training"
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug output")
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        config = bt.config(parser)

        if config.debug:
            tplr.debug()

        return config

    def __init__(self):
        # Initialize config
        self.config = self.agg_config()
        self.hparams = tplr.load_hparams()
        self.version = tplr.__version__
        self.wallet = bt.wallet(config=self.config)

        # Initialize bittensor objects
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(cast(int, self.config.netuid))
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Initialize model for gradient processing
        self.model_config = LlamaConfig(
            hidden_size=2048,
            num_hidden_layers=16,
            num_attention_heads=8,
            intermediate_size=8192,
            num_key_value_heads=8,
            max_position_embeddings=2048,
        )
        self.model = LlamaForCausalLM(self.model_config)
        self.model.to(self.config.device)  # Keep on CPU to save memory

        # Initialize compression
        self.transformer = tplr.compress.TransformDCT(
            self.model, target_chunk=self.hparams.target_chunk
        )
        self.compressor = tplr.compress.CompressDCT()

        # Pre-calculate shapes and totalks for all parameters
        self.param_shapes = {}
        self.param_totalks = {}
        tplr.logger.info("Pre-calculating compression parameters...")
        for name, param in self.model.named_parameters():
            _, _, shape, totalk = self.compressor.compress(
                self.transformer.encode(param.data), topk=self.hparams.topk_compression
            )
            self.param_shapes[name] = shape
            self.param_totalks[name] = totalk
        tplr.logger.info("Pre-calculation complete.")

        # Initialize comms
        self.comms = tplr.comms.Comms(
            wallet=self.wallet,
            save_location="/tmp",
            key_prefix="aggregator",
            config=self.config,
            netuid=self.config.netuid,
            metagraph=self.metagraph,
            hparams=self.hparams,
            uid=self.uid,
        )
        self.comms.bucket = self.comms.get_own_bucket("aggregator", "write")

        # Initialize state
        self.current_block = self.subtensor.block
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)

        self.iteration_counter = 0

    async def get_current_window(
        self, wait_for_completion=True
    ) -> tuple[int, datetime, datetime]:
        """Get the current window and calculate time bounds."""
        # Get current block and window
        current_block = self.subtensor.block
        current_window = int(current_block / self.hparams.blocks_per_window)

        # Calculate when this window ends
        window_end_block = (current_window + 1) * self.hparams.blocks_per_window

        # If we're still in the window and should wait, wait for it to complete
        if current_block < window_end_block and wait_for_completion:
            tplr.logger.info(f"\nWaiting for window {current_window} to complete...")
            while self.subtensor.block < window_end_block:
                await asyncio.sleep(1)

            tplr.logger.info(f"Window {current_window} completed")

        # Get timestamp of sync block for time window calculation
        sync_block = (current_window + 1) * self.hparams.blocks_per_window
        try:
            # Get the timestamp from the blockchain
            timestamp = self.subtensor.query_module(
                "Timestamp", "Now", block=sync_block
            )
            if not isinstance(timestamp, ScaleObj):
                raise ValueError(f"Could not query timestamp for {sync_block}")
            time_min = datetime.fromtimestamp(
                cast(int, timestamp.value) / 1000, tz=timezone.utc
            )
            # Add time window delta
            time_window_delta_seconds = self.hparams.time_window_delta_seconds
            time_max = time_min + timedelta(seconds=time_window_delta_seconds)

            tplr.logger.info(
                f"Time window for gradient validation: {time_min} to {time_max}"
            )
        except Exception:
            tplr.logger.exception("Error getting timestamp")
            # Fallback - use current time with generous window
            time_min = datetime.now(timezone.utc) - timedelta(minutes=30)
            time_max = datetime.now(timezone.utc) + timedelta(minutes=30)
            tplr.logger.info(f"Using fallback time window: {time_min} to {time_max}")

        return current_window, time_min, time_max

    async def process_window(self):
        """Process a single window: gather gradients, aggregate, and store."""
        tplr.logger.info(
            f"Starting window processing (iteration {self.iteration_counter}, window {self.sync_window - 1})"
        )

        # Calculate time window for this window
        sync_block = (self.sync_window) * self.hparams.blocks_per_window
        try:
            # Get the timestamp from the blockchain
            timestamp = self.subtensor.query_module(
                "Timestamp", "Now", block=sync_block
            )
            if not isinstance(timestamp, ScaleObj):
                raise ValueError(f"Could not query timestamp for {sync_block}")
            time_min = datetime.fromtimestamp(
                cast(int, timestamp.value) / 1000, tz=timezone.utc
            )
            # Add time window delta
            time_window_delta_seconds = self.hparams.time_window_delta_seconds
            time_max = time_min + timedelta(seconds=time_window_delta_seconds)

            tplr.logger.info(
                f"Time window for gradient validation: {time_min} to {time_max}"
            )
        except Exception:
            tplr.logger.exception("Error getting timestamp")
            # Fallback - use current time with generous window
            time_min = datetime.now(timezone.utc) - timedelta(minutes=30)
            time_max = datetime.now(timezone.utc) + timedelta(minutes=30)
            tplr.logger.info(f"Using fallback time window: {time_min} to {time_max}")

        # Wait until t_max has passed, plus additional wait time
        now = datetime.now(timezone.utc)
        if now < time_max:
            wait_seconds = (time_max - now).total_seconds() + cast(
                int, self.config.wait_time
            )
            tplr.logger.info(
                f"Waiting {wait_seconds:.1f} seconds until after t_max plus {self.config.wait_time}s buffer..."
            )
            await asyncio.sleep(wait_seconds)
            tplr.logger.info(
                f"Wait complete. Starting gradient collection at {datetime.now(timezone.utc)}"
            )
        else:
            # We're already past t_max, just add a small buffer
            tplr.logger.info(
                f"Already past t_max. Waiting additional {self.config.wait_time}s buffer..."
            )
            await asyncio.sleep(cast(int, self.config.wait_time))
            tplr.logger.info(
                f"Wait complete. Starting gradient collection at {datetime.now(timezone.utc)}"
            )

        # Use comms to select gather peers
        await self.update_peers(self.sync_window - 1)
        selected_uids = self.comms.peers

        tplr.logger.info(
            f"\nSelected {len(selected_uids)} peers for gradient collection"
        )
        tplr.logger.info(
            f"Selection parameters: topk={self.hparams.topk_peers}%, min={self.hparams.minimum_peers}, max_topk={self.hparams.max_topk_peers}"
        )
        tplr.logger.info(f"Selected UIDs: {selected_uids}")

        # Use the gather function to collect gradients
        tplr.logger.info(
            f"Starting gather operation for window {self.sync_window - 1}..."
        )
        gather_start = time.time()

        # Use the comms gather function (similar to how the miner uses it)
        gather_result = await self.comms.gather(
            my_uid=self.comms.uid,
            uids=selected_uids,
            window=self.sync_window - 1,
            key="gradient",
            timeout=15,
            device="cpu",
            local=False,
            stale_retention=100,
            totalks=self.param_totalks,
            time_min=time_min,
            time_max=time_max,
        )

        gather_time = time.time() - gather_start

        if gather_result is None:
            tplr.logger.warning(
                f"Failed to gather gradients for window {self.sync_window - 1}"
            )
            return False

        tplr.logger.info(f"Gather completed in {gather_time:.2f} seconds")
        tplr.logger.info(f"Successful gathers: {gather_result.success_rate * 100:.2f}%")
        tplr.logger.info(f"Skipped UIDs: {gather_result.skipped_uids}")

        # Process gathered gradients
        process_start = time.time()
        processed_state_dict = {}

        try:
            for name, param in self.model.named_parameters():
                idxs_key = name + "idxs"
                vals_key = name + "vals"

                idxs = getattr(gather_result.state_dict, idxs_key, None)
                vals = getattr(gather_result.state_dict, vals_key, None)

                if idxs is not None and vals is not None:
                    # Ensure idx and val are lists of tensors
                    if not isinstance(idxs, (list, tuple)):
                        idxs = [idxs]
                    if not isinstance(vals, (list, tuple)):
                        vals = [vals]

                    # Use the compressor to decompress the gradients
                    decompressed = self.compressor.batch_decompress(
                        param,
                        idxs,
                        vals,
                        self.param_shapes[name],
                        self.param_totalks[name],
                    )

                    # Pack the decompressed gradient
                    processed_state_dict[name] = tplr.neurons.pack_binary_tensor(
                        self.transformer.decode(decompressed)
                        .sign()
                        .to(self.config.device),
                        device=self.config.device,
                    ).cpu()

            process_time = time.time() - process_start
            tplr.logger.info(f"Processed gradients in {process_time:.2f} seconds")

            # Store the aggregated gradients
            store_start = time.time()
            processed_state_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
            processed_state_dict["window"] = self.sync_window - 1
            processed_state_dict["version"] = self.version
            processed_state_dict["skipped_uids"] = gather_result.skipped_uids
            processed_state_dict["success_rate"] = gather_result.success_rate

            try:
                await self.comms.put(
                    state_dict=processed_state_dict,
                    uid=self.comms.uid,
                    window=self.sync_window - 1,
                    key="aggregator",
                    local=False,
                    stale_retention=100,
                )
            except Exception:
                tplr.logger.warning(
                    f"Failed to store aggregation for window {self.sync_window - 1}"
                )
                return False

            store_time = time.time() - store_start

            tplr.logger.info(
                f"Successfully stored aggregation for window {self.sync_window - 1} in {store_time:.2f} seconds"
            )

            # Print summary
            total_time = gather_time + process_time + store_time
            tplr.logger.info(f"Window: {self.sync_window - 1}")
            tplr.logger.info(f"Target UIDs: {selected_uids}")
            tplr.logger.info(
                f"Successful UIDs: {len(selected_uids) - len(gather_result.skipped_uids)}"
            )
            tplr.logger.info(f"Skipped UIDs: {gather_result.skipped_uids}")
            tplr.logger.info(f"Gather time: {gather_time:.2f} seconds")
            tplr.logger.info(f"Process time: {process_time:.2f} seconds")
            tplr.logger.info(f"Store time: {store_time:.2f} seconds")
            tplr.logger.info(f"Total time: {total_time:.2f} seconds")

            return True

        except Exception:
            tplr.logger.exception("Error processing gradients")
            import traceback

            traceback.print_exc()
            return False

    async def run(self):
        """Main loop to continuously process windows."""
        tplr.logger.info("Starting aggregation server...")

        # Start background block listener thread
        self.loop = asyncio.get_running_loop()
        self.stop_event = asyncio.Event()
        self.listener = threading.Thread(
            target=self.block_listener, args=(self.loop,), daemon=True
        ).start()

        # Initialize comms background tasks
        self.comms.start_background_tasks()

        # Set up sync window to track which windows we've processed
        self.sync_window = self.current_window
        self.comms.current_window = self.current_window

        tplr.logger.info(f"Starting with window {self.current_window}")

        while True:
            try:
                # Update commitments and peers for upcoming process
                self.comms.commitments = await self.comms.get_commitments()

                # Process the current window
                await self.process_window()

                # Sync metagraph
                subtensor_sync = bt.subtensor(config=self.config)
                await asyncio.to_thread(
                    lambda: self.metagraph.sync(subtensor=subtensor_sync)
                )

                # Force garbage collection every 10 iterations
                if self.iteration_counter % 10 == 0:
                    tplr.logger.info("Running garbage collection...")
                    gc.collect()

                # Wait for next window if needed (optional)
                tplr.logger.info(
                    f"Waiting for next window... (current: {self.current_window})"
                )
                while self.sync_window >= self.current_window:
                    await asyncio.sleep(0.1)
                self.sync_window = self.current_window
                self.iteration_counter += 1

            except KeyboardInterrupt:
                self.stop_event.set()
                tplr.logger.info("\nReceived keyboard interrupt. Exiting...")
                break
            except Exception:
                tplr.logger.exception("\n!!! Error in main processing loop")
                tplr.logger.info("Continuing to next window...")
                # Sleep for a bit to avoid tight loop in case of persistent errors
                await asyncio.sleep(30)

    # Listens for new blocks and sets self.current_block and self.current_window
    def block_listener(self, loop):
        import websockets.exceptions  # Ensure we catch websockets errors

        def handler(event):
            try:
                self.current_block = int(event["header"]["number"])
                new_window = int(self.current_block / self.hparams.blocks_per_window)
                if new_window != self.current_window:
                    self.current_window = new_window
                    self.comms.current_window = self.current_window
                    tplr.logger.info(
                        f"New block received. Current window updated to: {self.current_window}"
                    )
            except Exception as e:
                tplr.logger.error(f"Error processing block event: {e}")

        backoff = 1  # initial backoff in seconds
        max_backoff = 60  # maximum backoff limit

        while not self.stop_event.is_set():
            try:
                # This call subscribes to block headers and might throw keepalive errors
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(
                    handler
                )
                backoff = 1  # reset backoff if subscription exits without exception
            except websockets.exceptions.ConnectionClosedError as e:
                tplr.logger.warning(
                    f"Websocket ConnectionClosedError caught: {e}. Retrying in {backoff} seconds."
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
            except Exception as e:
                tplr.logger.error(
                    f"Block subscription error: {e}. Retrying in {backoff} seconds."
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    async def update_peers(self, window) -> None:
        # Get next peers
        if (
            self.next_peers is None  # next peers are not fetched yet
            and self.peers_update_window  # they should be on bucket by now
            + self.hparams.peer_replacement_frequency
            - window
            < self.hparams.peer_list_window_margin
        ):
            result = await self.comms.get_peer_list()
            if result is None:
                tplr.logger.info("Unable to get peer list from bucket")
            else:
                next_peers, peers_update_window = result
                tplr.logger.info(
                    f"Got peer list {next_peers} and update window "
                    f"{peers_update_window} from bucket"
                )
                if (
                    self.peers_update_window is None
                    or peers_update_window > self.peers_update_window
                ):
                    self.next_peers = next_peers
                    self.peers_update_window = peers_update_window
                    tplr.logger.info("This list is new, updating next_peers")

        # Update peers, if it's time
        if self.next_peers is not None and window >= self.peers_update_window:
            self.comms.peers = self.next_peers
            late_text = (
                f"{window - self.peers_update_window} windows late"
                if window - self.peers_update_window > 0
                else "on time"
            )
            tplr.logger.info(
                f"{window} Updated peers "
                f"{late_text} - gather:{len(self.comms.peers)}. Next update "
                f"expected on step window "
                f"{self.peers_update_window + self.hparams.peer_list_window_margin}"
            )
            self.next_peers = None
        else:
            reason = (
                "next peers are not defined yet"
                if self.next_peers is None
                else f"sync window is {window} and peers update window "
                f"is {self.peers_update_window}"
            )
            tplr.logger.info(f"Not time to replace peers yet: {reason}")


# Start the aggregation server
if __name__ == "__main__":
    asyncio.run(AggregationServer().run())
