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

import time


class CatchUpManager:
    def __init__(self, comms, hparams):
        """
        Initialize the catch-up manager.
        Args:
            comms: Comms instance used for underlying data gathering and gradient application.
            hparams: Hyperparameters including catch-up thresholds and batch settings.
        """
        self.comms = comms
        self.hparams = hparams
        self.logger = comms.logger

    async def perform_catchup(
        self,
        model,
        optimizer,
        scheduler,
        transformer,
        compressor,
        current_window: int,
        sync_window: int,
        device: str,
        peers: list,
        uid: str,
        global_step: int,
        totalks: dict,
    ) -> tuple:
        """
        Perform catch-up operations on missing windows.
        Returns:
            A tuple (success: bool, new_global_step, optimizer, scheduler)
        """
        window_gap = current_window - sync_window
        if window_gap <= self.hparams.catch_up_threshold:
            self.logger.info("No catch-up needed; window gap within threshold.")
            return False, global_step, optimizer, scheduler
        if len(peers) < self.hparams.catch_up_min_peers:
            self.logger.warning(f"Not enough peers ({len(peers)}) for catch-up.")
            return False, global_step, optimizer, scheduler

        catch_up_start = time.time()
        self.logger.info(f"Initiating catch-up for {window_gap} windows.")
        try:
            for start in range(
                sync_window + 1, current_window + 1, self.hparams.catch_up_batch_size
            ):
                batch_end = min(
                    start + self.hparams.catch_up_batch_size, current_window + 1
                )
                batch_windows = list(range(start, batch_end))
                if time.time() - catch_up_start > self.hparams.catch_up_timeout:
                    self.logger.warning("Catch-up exceeded maximum time.")
                    return False, global_step, optimizer, scheduler
                try:
                    gathered_data = await self.comms._gather_window_batch(
                        batch_windows=batch_windows,
                        uid=uid,
                        peers=peers,
                        device=device,
                        totalks=totalks,
                        global_step=global_step,
                    )
                    for w in sorted(gathered_data.keys()):
                        (
                            success,
                            new_global_step,
                        ) = await self.comms._apply_gathered_gradients(
                            gather_result=gathered_data[w],
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            transformer=transformer,
                            compressor=compressor,
                            device=device,
                            window=w,
                            global_step=global_step,
                        )
                        if success:
                            global_step = new_global_step
                except Exception as e:
                    self.logger.error(
                        f"Catch-up failed during batch {batch_windows}: {str(e)}"
                    )
                    return False, global_step, optimizer, scheduler
            return True, global_step, optimizer, scheduler
        except Exception as e:
            self.logger.error(f"Catch-up failed: {str(e)}")
            return False, global_step, optimizer, scheduler
