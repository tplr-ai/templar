import abc
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import signal
import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import bittensor as bt
from bittensor.core.subtensor import ScaleObj
import torch.distributed as dist
import websockets.exceptions  # ensure import before threads start

import tplr


class BaseNode:
    # ――― attributes that subclasses will overwrite ―――
    executor: ThreadPoolExecutor
    world_size: int = 1
    comms: tplr.Comms
    config: Any = None
    hparams: SimpleNamespace
    subtensor: bt.Subtensor

    # background bookkeeping
    stop_event: asyncio.Event
    _bg_tasks: set[asyncio.Task]

    def __init__(self):
        # -------- shared state ------------------------------------------------
        self.stop_event = asyncio.Event()
        self._bg_tasks: set[asyncio.Task] = set()
        self._threads: list[threading.Thread] = []

        # ---- things your subclasses already set -----------------------------
        self.current_block = 0
        self.current_window = 0
        # self.config, self.world_size …  come from the concrete node

    async def main(self):
        loop = asyncio.get_running_loop()
        self._setup_signal_handlers(loop)

        # start the block-listener *thread*
        t = threading.Thread(target=self.block_listener, name="blocks", daemon=True)
        t.start()
        self._threads.append(t)

        # subclasses do their normal work here ----------------------------
        try:
            await self.run()
        finally:
            # ensure we exit cleanly even if run() returns without CTRL-C
            if not self.stop_event.is_set():
                await self._graceful_shutdown(signal.SIGTERM)

    @abc.abstractmethod
    async def run(self):
        raise NotImplementedError

    def _setup_signal_handlers(self, loop: asyncio.AbstractEventLoop):
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                functools.partial(asyncio.create_task, self._graceful_shutdown(sig)),
            )

    async def _graceful_shutdown(self, sig):
        tplr.logger.warning(f"↩  received {sig.name} – shutting down …")
        self.stop_event.set()

        # cancel background asyncio tasks
        for task in list(self._bg_tasks):
            task.cancel()
        await asyncio.gather(*self._bg_tasks, return_exceptions=True)

        # wait for helper threads
        for th in self._threads:
            th.join(timeout=2)

        # external resources
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False, cancel_futures=True)
        if (
            hasattr(self, "world_size")
            and self.world_size > 1
            and dist.is_initialized()
        ):
            dist.destroy_process_group()

        await asyncio.sleep(0.2)  # allow log flush
        tplr.logger.info("✔ shutdown complete")

    def query_block_timestamp(
        self,
        block: int,
        *,
        retries: int = 5,
        init_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> float | None:
        """
        Return the Unix-epoch timestamp (seconds) for `block` or None.

        • Uses a *new* Subtensor rpc client each try, so there is
          no clash with the streaming websocket in `block_listener`.
        • Retries with exponential back-off.
        """
        delay = init_delay
        for attempt in range(1, retries + 1):
            try:
                resp = bt.subtensor(config=self.config).query_module(
                    "Timestamp", "Now", block=block
                )
                if resp is None or not isinstance(resp, ScaleObj):
                    raise ValueError(f"Could not query timestamp for {block}")
                ts_value = (
                    cast(int, resp.value) / 1000
                )  # convert milliseconds to seconds
                return ts_value
            except Exception as e:
                tplr.logger.warning(
                    f"[timestamp] block {block} attempt {attempt}/{retries} failed: {e}"
                )
                if attempt == retries:
                    return None
                time.sleep(delay)
                delay = min(delay * 2, max_delay)

    def block_listener(self):
        backoff, max_backoff = 1, 60

        def handler(event):
            try:
                self.current_block = int(event["header"]["number"])
                new_window = self.current_block // self.hparams.blocks_per_window
                if new_window != self.current_window:
                    self.current_window = new_window
                    if hasattr(self, "comms"):
                        self.comms.current_window = self.current_window
                    tplr.logger.info(f"▶ window → {self.current_window}")
            except Exception as e:
                tplr.logger.error(f"block-handler err: {e}")

        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(
                    handler
                )
                backoff = 1  # normal exit ⇒ reset back-off
            except websockets.exceptions.ConnectionClosedError as e:
                if self.stop_event.is_set():  # shutting down → leave loop
                    break
                tplr.logger.warning(f"ws closed: {e} – retrying in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
            except Exception as e:
                if self.stop_event.is_set():
                    break
                tplr.logger.error(
                    f"block subscription err: {e} – retrying in {backoff}s"
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
