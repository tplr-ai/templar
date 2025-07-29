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

import argparse
import asyncio
import concurrent.futures
import io
import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import cast

import bittensor as bt
import torch
import uvloop
from aiobotocore import session
from botocore import exceptions 

import tplr
from tplr import config, comms
from tplr.logging import logger

CPU_COUNT = os.cpu_count() or 4


class GradientCollector:
    # ───────────────────────── config helper ───────────────────────────────
    @staticmethod
    def collector_config():
        p = argparse.ArgumentParser(description="Gradient Collector")
        p.add_argument("--netuid", type=int, default=3, help="Bittensor network UID.")
        bt.subtensor.add_args(p)
        c = bt.config(p)
        if c.debug:
            tplr.debug()
        return c

    # ───────────────────────── constructor ─────────────────────────────────
    def __init__(self):
        # 1) static configuration ------------------------------------------------
        self.config = self.collector_config()
        self.hparams = tplr.load_hparams()
        self.version = tplr.__version__

        # 2) destination bucket credentials -------------------------------------
        self.bucket_name = os.getenv("R2_COLLECTOR_BUCKET_NAME", "gradients-bucket")
        self.bucket_region = os.getenv("R2_COLLECTOR_BUCKET_REGION", "enam")
        self.bucket_access_key = os.getenv("R2_COLLECTOR_WRITE_ACCESS_KEY_ID")
        self.bucket_secret_key = os.getenv("R2_COLLECTOR_WRITE_SECRET_ACCESS_KEY")
        if not all([self.bucket_access_key, self.bucket_secret_key]):
            logger.warning("R2 credentials not fully specified in env vars")

        # 3) bittensor objects ---------------------------------------------------
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(cast(int, self.config.netuid))

        # 4) comms (already async / aiobotocore) ---------------------------------
        self.comms = comms.Comms(
            wallet=None,
            key_prefix="collector",
            config=self.config,
            netuid=self.config.netuid,
            metagraph=self.metagraph,
            hparams=self.hparams,
        )

        # 5) we’ll open the aiobotocore client later inside the event-loop
        self.bucket = None  # type: ignore

        # 6) state trackers ------------------------------------------------------
        self.current_block = self.subtensor.block
        self.current_window = self.current_block // self.hparams.blocks_per_window
        self.iteration_counter = 0
        self.next_peers: comms.PeerArray | None = None
        self.peers_update_window = -1

    # ───────────────────────── bucket init (async) ─────────────────────────
    async def _open_bucket(self):
        """
        Create (and cache) an aiobotocore S3 client for our **destination**
        bucket.  Called once at the start of `.run()`.
        """
        if self.bucket:
            return self.bucket

        account_id = os.getenv("R2_COLLECTOR_ACCOUNT_ID")
        if not account_id:
            raise RuntimeError("R2_COLLECTOR_ACCOUNT_ID env var is required")

        endpoint = self.comms.get_base_url(account_id)
        session = session.get_session()
        self.bucket = await session.create_client(
            "s3",
            endpoint_url=endpoint,
            region_name=self.bucket_region,
            config=config.client_config,
            aws_access_key_id=self.bucket_access_key,
            aws_secret_access_key=self.bucket_secret_key,
        ).__aenter__()

        try:
            await self.bucket.head_bucket(Bucket=self.bucket_name)  # type: ignore [reportGeneralTypeIssues]
        except exceptions.ClientError:
            logger.info(f"Creating bucket '{self.bucket_name}' in R2")
            await self.bucket.create_bucket(Bucket=self.bucket_name)  # type: ignore [reportGeneralTypeIssues]

        return self.bucket

    # ───────────────────────── core loop helper ────────────────────────────
    async def collect_window_gradients(self):
        """Download each peer's gradient and upload it to our bucket."""
        target_window = self.sync_window - 2
        start_time = time.time()

        logger.info(
            f"Collecting window {target_window} – iteration {self.iteration_counter}"
        )

        # 1. peer list selection --------------------------------------------------
        await tplr.neurons.update_peers(
            instance=self,  # type: ignore [reportArgumentType]
            window=target_window,
            peer_start=tplr.T(),
        )
        selected_uids: list[int] = [int(u) for u in self.comms.peers]
        logger.info(f"Selected {len(selected_uids)} peers: {selected_uids}")

        bucket = await self._open_bucket()  # aiobotocore client
        semaphore = asyncio.Semaphore(15)  # limit concurrent S3 ops

        # Track stats for reporting
        successful_count = 0
        total_bytes = 0

        async def _process(uid: int) -> tuple[int, bool]:
            nonlocal successful_count, total_bytes
            process_start = time.time()
            try:
                # 2. download from peer ------------------------------------------
                download_start = time.time()
                gradient = await self.comms.get_with_retry(
                    uid=str(uid),
                    window=target_window,
                    key="gradient",
                    local=False,
                    stale_retention=100,
                    timeout=60,
                )
                download_time = time.time() - download_start

                if gradient is None:
                    logger.debug(f"UID {uid}: No gradient received")
                    return uid, False

                logger.debug(
                    f"UID {uid}: Downloaded gradient in {download_time:.2f}s"
                )

                # 3. wrap + upload ----------------------------------------------
                payload = {
                    "gradient": gradient,
                    "metadata": {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "window": target_window,
                        "version": self.version,
                        "uid": uid,
                        "block": self.current_block,
                    },
                }
                buffer = io.BytesIO()
                torch.save(payload, buffer)
                buffer.seek(0)
                data = buffer.getvalue()
                data_size = len(data)
                total_bytes += data_size

                upload_start = time.time()
                key = f"v{self.version}/{target_window}/{uid}/gradient.pt"
                async with semaphore:
                    await bucket.put_object(  # type: ignore [reportGeneralTypeIssues]
                        Bucket=self.bucket_name,
                        Key=key,
                        Body=data,
                    )
                upload_time = time.time() - upload_start

                total_time = time.time() - process_start
                logger.debug(
                    f"UID {uid}: Uploaded gradient ({data_size / 1024:.1f} KB) in {upload_time:.2f}s"
                )
                logger.info(
                    f"UID {uid}: Processed successfully in {total_time:.2f}s"
                )

                successful_count += 1
                return uid, True

            except (exceptions.ConnectionClosedError, exceptions.ClientError) as e:
                logger.debug(f"S3 error for UID {uid}: {e}")
                return uid, False
            except Exception as e:
                logger.warning(f"UID {uid}: {e}")
                return uid, False

        # run all in parallel -----------------------------------------------------
        gather_start = time.time()
        results = await asyncio.gather(*[_process(u) for u in selected_uids])
        gather_time = time.time() - gather_start

        ok_uids = [u for u, ok in results if ok]
        bad_uids = [u for u, ok in results if not ok]

        # 4. write summary --------------------------------------------------------
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "window": target_window,
            "version": self.version,
            "selected_uids": selected_uids,
            "successful_uids": ok_uids,
            "failed_uids": bad_uids,
            "success_rate": len(ok_uids) / len(selected_uids) if selected_uids else 0.0,
            "block": self.current_block,
            "gather_time_seconds": gather_time,
            "total_size_mb": total_bytes / (1024 * 1024),
        }
        await bucket.put_object(  # type: ignore [reportGeneralTypeIssues]
            Bucket=self.bucket_name,
            Key=f"v{self.version}/{target_window}/summary.json",
            Body=json.dumps(summary).encode(),
        )

        total_time = time.time() - start_time

        # Log comprehensive summary
        logger.info(
            f"Window {target_window}: success {len(ok_uids)}/{len(selected_uids)} "
            f"({summary['success_rate'] * 100:.1f}%)"
        )
        logger.info(f"Total gather time: {gather_time:.2f}s")
        logger.info(f"Total execution time: {total_time:.2f}s")
        logger.info(f"Total data collected: {total_bytes / 1024 / 1024:.2f} MB")

        if bad_uids:
            logger.debug(f"Failed UIDs: {bad_uids}")

        return True

    # ───────────────────────── main run-loop ───────────────────────────────
    async def run(self):
        logger.info("Gradient collector starting…")
        await self._open_bucket()

        # thread for block-headers updates ----------------------------------------
        self.loop = asyncio.get_running_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
        self.loop.set_default_executor(self.executor)

        self.stop_event = asyncio.Event()
        threading.Thread(
            target=self.block_listener, args=(self.loop,), daemon=True
        ).start()

        # sync-window bookkeeping -------------------------------------------------
        self.sync_window = self.current_window
        self.comms.current_window = self.current_window
        self.comms.commitments = await self.comms.get_commitments()
        self.start_window = await self.comms.get_start_window()

        while True:
            try:
                self.comms.commitments = await self.comms.get_commitments()
                await self.collect_window_gradients()

                # refresh metagraph once a cycle
                sync = bt.subtensor(config=self.config)
                await asyncio.to_thread(lambda: self.metagraph.sync(subtensor=sync))

                # wait for next window
                while self.sync_window >= self.current_window:
                    await asyncio.sleep(1)
                self.sync_window = self.current_window
                self.iteration_counter += 1

            except KeyboardInterrupt:
                self.stop_event.set()
                break
            except Exception as e:
                logger.exception(f"Main loop error: {e}")
                await asyncio.sleep(30)

    # ───────────────────────── block listener  ─────────────────────────
    def block_listener(self, loop):
        from websockets import exceptions

        def handler(event):
            try:
                self.current_block = int(event["header"]["number"])
                window = self.current_block // self.hparams.blocks_per_window
                if window != self.current_window:
                    self.current_window = window
                    self.comms.current_window = window
                    logger.info(f"→ new window {window}")
            except Exception as e:
                logger.error(f"block_listener: {e}")

        backoff, max_backoff = 1, 60
        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(
                    handler
                )
                backoff = 1
            except exceptions.ConnectionClosedError:
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
            except Exception as e:
                logger.warning(f"block_listener: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(GradientCollector().run())
