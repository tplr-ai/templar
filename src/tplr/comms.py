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
# type: ignore
import asyncio
import concurrent.futures
import json
import math
import os
import random
import re
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial

# from .hparams import HParams
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional

import aiofiles
import bittensor as bt
import botocore
import torch
from aiobotocore.client import AioBaseClient
from aiobotocore.session import get_session
from botocore.exceptions import ClientError, ConnectionClosedError
from tqdm import tqdm as std_tqdm

import tplr
from tplr import __version__
from tplr.chain import ChainManager
from tplr.compress import TopKCompressor, unpack_12bit_indices
from tplr.config import BUCKET_SECRETS, client_config
from tplr.io import CF_REGION_NAME, S3Manager, get_own_bucket, async_s3_exception_catcher
from tplr.schemas import Bucket, CommsGetResult

# Constants
LOCAL_TMP_DIR = "/tmp"
PEERS_FILE_PREFIX = "peers_"
CPU_COUNT = os.cpu_count() or 4
CPU_MAX_CONNECTIONS = min(100, max(30, CPU_COUNT * 4))


class Comms(ChainManager, S3Manager):
    """Manage miner/vali communications"""

    def __init__(
        self,
        wallet: bt.wallet | None,
        key_prefix: str = "model",
        config=None,
        netuid=None,
        metagraph=None,
        hparams=None,
        uid=None,
    ):
        self.uid = uid
        self.wallet = wallet

        # # Create temp directory for this instance
        # self.temp_dir = os.path.join("/tmp", f"templar_{self.uid}")
        # os.makedirs(self.temp_dir, exist_ok=True)

        self.bucket = get_own_bucket("gradients", "write")

        # Now initialize ChainManager with the bucket
        super().__init__(
            config=config,
            netuid=netuid,
            metagraph=metagraph,
            hparams=hparams,
            wallet=self.wallet,
            bucket=self.bucket,
        )

        self.s3_manager = S3Manager(
            # wallet,
            # key_prefix,
            # config,
            # netuid,
            # metagraph,
            # hparams,
            # uid,
            save_location=LOCAL_TMP_DIR,
        )

        # Set base location
        self.save_location = LOCAL_TMP_DIR
        # Use the hotkey directly in the save_location
        if self.wallet is not None:
            hotkey = self.wallet.hotkey.ss58_address
            self.save_location = os.path.join("/tmp", f"hotkey_{hotkey}")
        os.makedirs(self.save_location, exist_ok=True)
        self.key_prefix = key_prefix

        ## a single aiobotocore session and a dictionary of clients
        # self.session = get_session()

        self.lock = asyncio.Lock()
        self.active_peers = set()  # Set to store active peers
        self.active_check_interval = (
            self.hparams.active_check_interval
        )  # Interval in seconds
        self.recent_windows = (
            self.hparams.recent_windows
        )  # Number of recent windows to check
        self.peers: list[int] = []
        self.reserve_peers: list[int] = []

        # self.client_semaphore = asyncio.Semaphore(CPU_MAX_CONNECTIONS)
        self.gather_semaphore = asyncio.Semaphore(20)

    def start_background_tasks(self):
        """Enable threadpooling"""
        self.loop = asyncio.get_running_loop()

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)
        self.loop.set_default_executor(self.executor)

        # Start background tasks
        self.loop.create_task(self.track_active_peers())
    
    @async_s3_exception_catcher(on_error_return=0.0)
    async def gradient_timestamp(
        self, uid: int, window: int, version: str = tplr.__version__
    ) -> float:
        """
        Return POSIX seconds of the gradient file’s Last-Modified header,
        or 0.0 if it does not exist / fails.
        """
        bucket = self.commitments.get(int(uid))
        # if not bucket: # handled by wrapper
        #     return 0.0
        
        s3 = await self._get_s3_client(bucket)
        key = f"gradient-{window}-{uid}-v{version}.pt"
        hdr = await s3.head_object(Bucket=bucket.name, Key=key)
        return hdr["LastModified"].timestamp()
    
    @async_s3_exception_catcher(on_error_return=CommsGetResult(status="ERROR"))
    async def get(
        self,
        uid: str,
        window: int,
        key: Literal["checkpoint", "debug", "gradient", "aggregator"],
        local: bool = True,
        stale_retention: int = 10,
        timeout: int = 30,
        time_min: datetime = None,
        time_max: datetime = None,
    ) -> CommsGetResult:
        """GET operation."""
        if key == "aggregator":
            filename = f"{key}-{window}-v{__version__}.pt"
        else:
            filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        tplr.logger.debug(f"GET {filename} -->")

        try:
            if local:
                # Local storage logic
                await self.cleanup_local_data(
                    uid=uid, current_window=window, stale_retention=stale_retention
                )
                local_path = os.path.join(
                    LOCAL_TMP_DIR, str(uid), str(window), filename
                )
                if not os.path.exists(local_path):
                    tplr.logger.debug(f"Local file not found: {local_path}")
                    return CommsGetResult(status="NOT_FOUND")
                loaded_data = torch.load(local_path, weights_only=True)
                if key == "checkpoint":
                    return CommsGetResult(data=loaded_data)
                state_dict = loaded_data.get("state_dict")
                global_step = loaded_data.get("global_step", 0)
                return CommsGetResult(data=state_dict, global_step=global_step)

            # Remote storage logic
            if key == "aggregator":
                bucket_config = BUCKET_SECRETS["aggregator"]
                credentials = bucket_config["credentials"]["read"]

                # Create a Bucket object using specified credentials
                bucket = Bucket(
                    name=bucket_config["name"],
                    account_id=bucket_config["account_id"],
                    access_key_id=credentials["access_key_id"],
                    secret_access_key=credentials["secret_access_key"],
                )
            else:
                bucket = self.commitments.get(int(uid))
            tplr.logger.debug(f"Peer bucket : {bucket}")
            if not bucket:
                return CommsGetResult(status="NOT_FOUND")

            loaded_data = await self.s3_get_object(
                key=filename,
                bucket=bucket,
                timeout=timeout,
                time_min=time_min,
                time_max=time_max,
            )

            if loaded_data is None:
                return CommsGetResult(status="NOT_FOUND")

            # Check for TOO_LATE/TOO_EARLY marker
            if isinstance(loaded_data, dict):
                if loaded_data.get("__status") == "TOO_LATE":
                    tplr.logger.info(
                        f"Object for UID {uid}, window {window}, key {key} was uploaded too late. Skipping."
                    )
                    return CommsGetResult(status="TOO_LATE")
                elif loaded_data.get("__status") == "TOO_EARLY":
                    tplr.logger.info(
                        f"Object for UID {uid}, window {window}, key {key} was uploaded too early. Skipping."
                    )
                    return CommsGetResult(status="TOO_EARLY")

            if key == "checkpoint":
                return CommsGetResult(data=loaded_data)

            state_dict = loaded_data.get("state_dict")
            global_step = loaded_data.get("global_step", 0)
            return CommsGetResult(data=state_dict, global_step=global_step)

        finally:
            tplr.logger.debug(f"GET {filename} <--")

    async def get_with_retry(
        self,
        uid: str,
        window: int,
        key: str,
        timeout: int,
        local: bool = True,
        stale_retention: int = 10,
        time_min: datetime = None,
        time_max: datetime = None,
    ) -> Optional[tuple[dict, int]]:
        """GET with retry operation."""
        start_time = time.time()
        end_time = start_time + timeout
        tried_after_time_max = False
        time_max_grace_period = 3.0

        while True:
            # Check if we've timed out
            if time.time() >= end_time:
                tplr.logger.debug(f"GET {uid}/{window}/{key} timed out.")
                return None

            # Check if we're past time_max with grace period
            now = datetime.now(timezone.utc)

            # Only consider it "past time_max" if we're 3 seconds beyond time_max
            past_time_max = False
            if time_max is not None and now > time_max:
                seconds_past_time_max = (now - time_max).total_seconds()
                past_time_max = seconds_past_time_max > time_max_grace_period

            # If we're past time_max (with grace period) and already tried once, don't retry again
            if past_time_max and tried_after_time_max:
                tplr.logger.debug(
                    f"Already tried once after time_max + {time_max_grace_period}s for UID {uid}, window {window}. Stopping retries."
                )
                return None

            # If we're past time_max (with grace period), mark that we've tried once
            if past_time_max:
                tried_after_time_max = True
                tplr.logger.debug(
                    f"Past time_max + {time_max_grace_period}s for UID {uid}, window {window}. This is the final retry."
                )

            # Make the request
            result = await self.get(
                uid=uid,
                window=window,
                key=key,
                local=local,
                stale_retention=stale_retention,
                time_min=time_min,
                time_max=time_max,
            )

            if result.success:
                return result.data, result.global_step

            if result.status == "TOO_LATE":
                tplr.logger.info(
                    f"Gradient for UID {uid}, window {window} exists but was uploaded too late. Skipping."
                )
                return None

            # For NOT_FOUND, ERROR, or TOO_EARLY, we retry.
            # Short delay before retrying
            await asyncio.sleep(0.5)

    async def gather(
        self,
        my_uid: int | None,
        uids: list[int],
        window: int,
        key: str,
        timeout: int,
        device: str,
        totalks: dict,
        compressor: TopKCompressor,
        expected_compressed_params: set[str] | None = None,
        local: bool = True,
        stale_retention: int = 10,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
    ) -> SimpleNamespace | None:
        """Gather operation with individual gradient normalization and connection management."""
        if not expected_compressed_params:
            expected_compressed_params = set()

        start_time = time.time()
        metrics = {"upload_bytes": 0, "download_bytes": 0, "successes": []}

        tplr.logger.debug(
            f"Starting gather for window {window} with time window: {time_min} to {time_max}"
        )
        tplr.logger.debug(
            f"Gather operation - my_uid: {my_uid}, window: {window}, key: {key}, timeout: {timeout}"
        )
        tplr.log_with_context(
            level="debug",
            message=f"Target UIDs for gathering: {uids}",
            current_window=window,
        )

        aggregated_state_dict = {}
        valid_uids = []
        skipped_uids = []  # Retain UIDs that are skipped.
        global_steps = []

        async with self.gather_semaphore:
            batch_tasks = [
                self.get_with_retry(
                    uid=uid,
                    window=window,
                    key=key,
                    timeout=timeout,
                    local=local,
                    stale_retention=stale_retention,
                    time_min=time_min,
                    time_max=time_max,
                )
                for uid in uids
            ]

            try:
                download_start = tplr.T()
                batch_responses = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )
                tplr.logger.info(
                    f"{tplr.P(window, tplr.T() - download_start)} Downloaded peer gradients <--"
                )
                process_start = tplr.T()
                for uid, response in zip(uids, batch_responses):
                    received_compressed_params = set()

                    if isinstance(response, Exception):
                        tplr.log_with_context(
                            level="debug",
                            message=f"Error from UID {uid}: {str(response)}",
                            current_window=window,
                        )
                        skipped_uids.append(uid)
                        continue
                    if response is None:
                        tplr.logger.info(f"Skipped UID {uid} - gradient not found.")
                        skipped_uids.append(uid)
                        continue

                    try:
                        state_dict_resp, global_step_resp = response
                        tplr.logger.debug(
                            f"Received state dict and global step {global_step_resp} from UID {uid}"
                        )
                    except (TypeError, ValueError) as e:
                        tplr.log_with_context(
                            level="debug",
                            message=f"Invalid response from UID {uid}: {e}",
                            current_window=window,
                        )
                        skipped_uids.append(uid)
                        continue

                    if state_dict_resp is None:
                        tplr.logger.debug(f"Empty state dict from UID {uid}")
                        skipped_uids.append(uid)
                        continue

                    decoded_cache: dict[str, torch.Tensor] = {}

                    # ---------- Begin Compressed Indices and Values Check ----------
                    valid_response = True
                    for param_name, tensor in state_dict_resp.items():
                        received_compressed_params.add(param_name)

                        # ----------------------------------------------------------
                        # (1)  Validate quantisation parameters themselves
                        # ----------------------------------------------------------
                        if param_name.endswith("quant_params"):
                            shift, scale, offset, lookup, dtype = tensor
                            if (
                                (not torch.isfinite(shift))
                                or isinstance(scale, float)
                                and (
                                    not math.isfinite(scale)
                                    or abs(scale) < 1e-12
                                    or abs(scale) > 1e4
                                )
                            ):
                                tplr.logger.warning(
                                    f"Bad quant‑params in {param_name} from UID {uid}; "
                                    f"shift={shift}, scale={scale}"
                                )
                                valid_response = False
                                break
                            if torch.is_tensor(lookup) and (
                                not torch.isfinite(lookup).all()
                            ):
                                tplr.logger.warning(
                                    f"Lookup table contains non‑finite values in {param_name} "
                                    f"from UID {uid}"
                                )
                                valid_response = False
                                break

                        if param_name.endswith("idxs"):
                            base_name = param_name[:-4]
                            totalk = totalks.get(base_name)
                            if totalk is None:
                                tplr.logger.warning(
                                    f"Missing totalk for parameter {base_name} from UID {uid}, skipping UID."
                                )
                                valid_response = False
                                break
                            # Get corresponding vals tensor for 12-bit unpacking
                            vals_tensor = state_dict_resp.get(base_name + "vals", None)
                            try:
                                self.check_compressed_indices(
                                    param_name,
                                    tensor,
                                    totalk,
                                    allowed_topk=self.hparams.topk_compression,
                                    vals=vals_tensor,
                                )
                            except Exception as e:
                                tplr.logger.warning(
                                    f"Compressed indices check failed for parameter {param_name} from UID {uid}: {e}"
                                )
                                valid_response = False
                                break
                        # Check if values are valid (not NaN, not Inf)
                        elif param_name.endswith("vals"):
                            tensor_to_check = tensor.to(device)
                            if (
                                torch.isnan(tensor_to_check).any()
                                or torch.isinf(tensor_to_check).any()
                            ):
                                tplr.logger.warning(
                                    f"NaN/Inf in {param_name} from UID {uid}, skipping"
                                )
                                valid_response = False
                                break

                            # ------------------------------------------------------
                            # (2)  De‑quantise *just for validation* (cheap‑ish)
                            # ------------------------------------------------------
                            qparams = state_dict_resp.get(
                                param_name[:-4] + "quant_params", None
                            )
                            if qparams is not None:
                                try:
                                    vals_f32 = compressor._dequantize_values(
                                        tensor_to_check, qparams
                                    )
                                    if (
                                        not torch.isfinite(vals_f32).all()
                                    ) or vals_f32.abs().max() > 1e3:
                                        tplr.logger.warning(
                                            f"Decoded values in {param_name} from UID {uid} "
                                            f"are non‑finite or too large; max={vals_f32.abs().max()}"
                                        )
                                        valid_response = False
                                        break
                                    decoded_cache[param_name] = vals_f32
                                except Exception as e:
                                    tplr.logger.warning(
                                        f"De‑quantisation failed for {param_name} from UID {uid}: {e}"
                                    )
                                    valid_response = False
                                    break

                    missing_params = (
                        expected_compressed_params - received_compressed_params
                    )
                    if missing_params:
                        tplr.logger.warning(
                            f"UID {uid} missing compressed parameters: {missing_params}, skipping UID."
                        )
                        valid_response = False

                    # If any check failed, skip this UID entirely
                    if not valid_response:
                        tplr.logger.info(
                            f"Skipping UID {uid} due to validation failures"
                        )
                        skipped_uids.append(uid)
                        continue
                    # ---------- End Compressed Indices and Values Check ----------

                    # Process tensors (with normalization on 'vals' keys).
                    for param_name, tensor in state_dict_resp.items():
                        # 1️⃣  Indices are kept as‑is -----------------------------------------
                        if param_name.endswith("idxs"):
                            aggregated_state_dict.setdefault(param_name, []).append(
                                tensor
                            )
                            # Handle 12-bit packed format (uint8 tensor)
                            metrics["download_bytes"] += (
                                tensor.element_size() * tensor.nelement()
                            )

                        # 2️⃣  Values → de‑quantise once and store as fp32 --------------------
                        elif param_name.endswith("vals"):
                            # Re-use if we already decoded during validation
                            tensor = decoded_cache.get(param_name, tensor.to(device))

                            # If still uint8 it means we skipped validation (unlikely),
                            # so decode now.
                            if tensor.dtype == torch.uint8:
                                qparams = state_dict_resp.get(
                                    param_name[:-4] + "quant_params", None
                                )
                                if qparams is not None:
                                    tensor = compressor._dequantize_values(
                                        tensor, qparams
                                    )

                            aggregated_state_dict.setdefault(param_name, []).append(
                                tensor
                            )
                            metrics["download_bytes"] += (
                                tensor.element_size() * tensor.nelement()
                            )

                    valid_uids.append(uid)
                    global_steps.append(global_step_resp)

                tplr.logger.info(
                    f"{tplr.P(window, tplr.T() - process_start)} Processed peer gradients <--"
                )

            except Exception as e:
                tplr.logger.error(f"Error processing uid batch: {str(e)}")

        if not valid_uids:
            tplr.logger.info("No valid gradients received from any UID")
            return None

        total_time = time.time() - start_time
        tplr.logger.info(
            f"Gather done in {total_time:.2f}s. Success rate: {len(valid_uids)}/{len(uids)}, "
            f"Upload: {metrics['upload_bytes']} bytes, Download: {metrics['download_bytes']} bytes"
        )

        result = SimpleNamespace(
            time=total_time,
            upload_bytes=metrics["upload_bytes"],
            download_bytes=metrics["download_bytes"],
            success_rate=len(valid_uids) / len(uids),
            state_dict=SimpleNamespace(**aggregated_state_dict),
            uids=valid_uids,
            global_steps=global_steps,
            skipped_uids=skipped_uids,
        )
        return result

    # ------------------------------------------------------------------
    # gather_with_reserve –– retry skipped gather peers with reserve tier
    # ------------------------------------------------------------------
    async def gather_with_reserve(
        self,
        *,
        my_uid: int | None,
        gather_uids: list[int],
        reserve_uids: list[int],
        expected_compressed_params: set[str] | None = None,
        **kwargs,
    ) -> SimpleNamespace | None:
        """
        1. Call `gather()` on the main `gather_uids`.
        2. Any UID that fails (or is missing) is replaced *once* with the next
           UID(s) from `reserve_uids`, then gathered again.
        3. Results are *merged* so the caller receives a single object that
           looks exactly like the old `gather()` return value.
        """
        if len(gather_uids + reserve_uids) == 0:
            return None

        if not expected_compressed_params:
            expected_compressed_params = set()

        window = kwargs.get("window", None)  # for contextual logs
        context_log = partial(tplr.log_with_context, level="info", window=window)

        context_log(
            message=f"[gather_with_reserve] ⏩ start | "
            f"gather={gather_uids} reserve={reserve_uids}"
        )

        primary = await self.gather(
            my_uid=my_uid,
            uids=gather_uids,
            expected_compressed_params=expected_compressed_params,
            **kwargs,
        )

        # Normalise to an empty shell if absolutely nothing came back
        if primary is None:
            primary = SimpleNamespace(
                time=0.0,
                upload_bytes=0,
                download_bytes=0,
                success_rate=0.0,
                state_dict=SimpleNamespace(),
                uids=[],
                global_steps=[],
                skipped_uids=gather_uids.copy(),
            )

        context_log(
            message=f"[gather_with_reserve] ✅ primary gather "
            f"{len(primary.uids)}/{len(gather_uids)} succeeded | "
            f"skipped={primary.skipped_uids}"
        )

        # ── 2. Retry the misses with reserve peers ─────────────────────
        missing = set(gather_uids) - set(primary.uids)
        if missing and reserve_uids:
            # take as many reserve peers as slots we missed
            replacements = [uid for uid in reserve_uids if uid not in primary.uids][
                : len(missing)
            ]

            if replacements:
                context_log(
                    message=f"[gather_with_reserve] 🔄 retrying with reserve "
                    f"uids={replacements}"
                )
                fallback = await self.gather(my_uid=my_uid, uids=replacements, **kwargs)
                if fallback:
                    # merge tensor‑lists inside the nested state_dict
                    for k, v in vars(fallback.state_dict).items():
                        merged = getattr(primary.state_dict, k, []) + v
                        setattr(primary.state_dict, k, merged)

                    primary.uids.extend(fallback.uids)
                    primary.global_steps.extend(fallback.global_steps)
                    primary.skipped_uids.extend(fallback.skipped_uids)
                    primary.upload_bytes += fallback.upload_bytes
                    primary.download_bytes += fallback.download_bytes

                    context_log(
                        message=f"[gather_with_reserve] ✅ reserve gather "
                        f"{len(fallback.uids)}/{len(replacements)} "
                        f"succeeded | skipped={fallback.skipped_uids}"
                    )

        # recompute success‑rate with respect to the *original* gather tier
        target = len(gather_uids)
        primary.success_rate = len(primary.uids) / target if target else 0.0

        context_log(
            message=f"[gather_with_reserve] 🏁 done | "
            f"final_success={len(primary.uids)}/{target} "
            f"({primary.success_rate:.1%}) | total_skipped={primary.skipped_uids}"
        )
        return primary

    ## Peer Management
    @async_s3_exception_catcher(on_error_return=False)
    async def is_miner_active(self, uid: int, recent_windows: int = 3) -> bool:
        """Check if the miner has uploaded gradients in the last few windows."""
        tplr.logger.debug(f"Checking if UID {uid} is active")
        current_window = self.current_window

        peer_bucket = self.commitments.get(uid)
        if not peer_bucket:
            tplr.log_with_context(
                level="debug",
                message=f"No bucket committed for UID {uid}",
                current_window=self.current_window,
            )
            return False

        s3_client = await self._get_s3_client(peer_bucket)

        if not hasattr(self, "current_window") or self.current_window is None:
            tplr.logger.error(
                "current_window is not set in comms. Please set comms.current_window."
            )
            return False

        current_window = self.current_window
        for window in range(current_window - recent_windows, current_window + 1):
            filename = f"gradient-{window}-{uid}-v{__version__}.pt"
            tplr.logger.debug(f"Checking for {filename} in {peer_bucket.name}")
            await s3_client.head_object(Bucket=peer_bucket.name, Key=filename)
            tplr.logger.debug(f"Found {filename} for UID {uid}")
            return True

        return False

    async def track_active_peers(self):
        """Background task to keep track of active peers."""
        while True:
            active_peers = set()
            max_concurrent = min(30, len(self.commitments) if self.commitments else 10)
            semaphore = asyncio.Semaphore(max_concurrent)

            tplr.logger.debug(f"Commitments: {self.commitments}")

            async def check_peer(uid):
                async with semaphore:
                    is_active = await self.is_miner_active(
                        uid, recent_windows=self.recent_windows
                    )
                    if is_active:
                        active_peers.add(uid)

            # Buffer the processing of commitments to avoid blocking the event loop (over resourcing)
            batch_size = 50
            commitment_uids = list(self.commitments.keys())

            for i in range(0, len(commitment_uids), batch_size):
                batch_uids = commitment_uids[i : i + batch_size]
                batch_tasks = [check_peer(uid) for uid in batch_uids]
                await asyncio.gather(*batch_tasks)

            self.active_peers = active_peers

            tplr.logger.info(
                f"Updated active peers: {[int(uid) for uid in self.active_peers]}"
            )

            await asyncio.sleep(self.active_check_interval)

    # Checkpoint Operations

    async def _get_highest_stake_validator_bucket(self):
        """Get the bucket for the validator with highest stake."""
        # Get validator with highest stake
        validator_uid = self.metagraph.S.argmax().item()
        tplr.logger.info(f"Found validator with highest stake: {validator_uid}")

        if validator_uid is None:
            tplr.logger.info("No active validators found")
            return None, None

        validator_bucket = self.commitments.get(int(validator_uid))
        if not validator_bucket:
            return None, None

        tplr.logger.info(f"Validator Bucket: {validator_bucket}")
        return validator_bucket, validator_uid

    @async_s3_exception_catcher
    async def get_latest_checkpoint(self, version):
        """
        Sequentially check:
        1. Whether the highest-staked validator has a checkpoint.
        2. Whether the R2 bucket of this instance has a checkpoint.
        3. Whether a checkpoint exists locally.
        If none are found, return None.
        """
        # 1. Check validator bucket
        (
            validator_bucket,
            validator_uid,
        ) = await self._get_highest_stake_validator_bucket()
        if validator_bucket:
            result = await self._get_bucket_checkpoint(
                validator_bucket, validator_uid, version
            )
            if result:
                # If successfully retrieved, return immediately.
                return result

        # 2. Check self R2 bucket
        self_bucket = self.bucket  # Use self.bucket saved in __init__
        if self_bucket:
            result = await self._get_bucket_checkpoint(
                self_bucket, self.uid, version
            )
            if result:
                return result

        # 3. Check local storage
        local_result = self._load_latest_local_checkpoint(version)
        if local_result:
            return local_result

        tplr.logger.info(
            "No checkpoint found in validator / self R2 / local storage"
        )
        return None

    @async_s3_exception_catcher
    def _load_latest_local_checkpoint(self, version: str):
        local_dir = os.path.join(LOCAL_TMP_DIR, str(self.uid))
        pattern = rf"checkpoint-(\d+)-{self.uid}-v{re.escape(version)}\.pt$"

        if not os.path.exists(local_dir):
            return None

        checkpoints = []
        for window_dir in os.listdir(local_dir):
            path = os.path.join(local_dir, window_dir)
            if not os.path.isdir(path):
                continue

            for file in os.listdir(path):
                match = re.match(pattern, file)
                if match:
                    # window number comes from match.group(1)
                    w = int(match.group(1))
                    file_path = os.path.join(path, file)
                    checkpoints.append(
                        {
                            "path": file_path,
                            "window": w,
                            "modified": os.path.getmtime(file_path),
                        }
                    )

        if checkpoints:
            # choose the last modified checkpoint
            latest = max(checkpoints, key=lambda x: x["modified"])
            checkpoint_data = torch.load(latest["path"], weights_only=True)
            return checkpoint_data, latest["window"]
        else:
            return None

    @async_s3_exception_catcher
    async def _get_bucket_checkpoint(self, bucket, uid, version: str):
        """Helper to get checkpoint from a specific bucket."""
        s3_client = await self._get_s3_client(bucket)

        pat = re.compile(rf"^checkpoint-(\d+)-{uid}-v{re.escape(version)}\.pt$")

        # We'll track the largest checkpoint window and its key
        latest_checkpoint = None
        max_window = -1

        # Continuation token for pagination
        continuation_token = None

        while True:
            list_kwargs = {
                "Bucket": bucket.name,
                "Prefix": "checkpoint",
            }
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token

            response = await s3_client.list_objects_v2(**list_kwargs)

            # If no objects returned, stop checking
            if not response.get("Contents"):
                break

            # Iterate through returned objects to find valid checkpoints
            for obj in response["Contents"]:
                key = obj.get("Key", "")
                match = pat.match(key)
                if match:
                    window_number = int(match.group(1))
                    if window_number > max_window:
                        max_window = window_number
                        latest_checkpoint = key

            # Continue pagination if needed
            if response.get("IsTruncated"):
                continuation_token = response.get("NextContinuationToken")
            else:
                # No more pages
                break

        # If we found a valid checkpoint, fetch it
        if latest_checkpoint:
            loaded_data = await self.s3_get_object(
                key=latest_checkpoint, bucket=bucket
            )
            if loaded_data:
                return loaded_data, max_window

        return None

    async def load_checkpoint(
        self,
        model,
        current_window: int,
        device: str,
        init_version: Optional[str] = None,
    ) -> tuple[bool, int]:
        """
        Loads the latest checkpoint. No catchup or step simulation happens here.
        Returns:
            tuple: (success: bool, checkpoint_current_window: int)
        """
        init_version = init_version if init_version is not None else __version__
        result = await self.get_latest_checkpoint(init_version)
        if not result:
            tplr.logger.info("No valid checkpoints found")
            return False, 0

        checkpoint_data, checkpoint_window = result
        try:
            # 1) Load model and optimizer state
            model.load_state_dict(
                {
                    k: v.to(device)
                    for k, v in checkpoint_data["model_state_dict"].items()
                }
            )
            model.to(device)

            checkpoint_start_window = checkpoint_data.get("start_window")
            checkpoint_current_window = checkpoint_data.get("current_window")
            checkpoint_sync_window = checkpoint_data.get("sync_window")
            if checkpoint_start_window is None or checkpoint_current_window is None:
                tplr.logger.warning(
                    "Checkpoint missing start_window or current_window info"
                )
                return False, 0

            tplr.logger.info(
                f"Checkpoint loaded. start_window={checkpoint_start_window}, "
                f"checkpoint_current_window={checkpoint_current_window}, "
                f"checkpoint_sync_window={checkpoint_sync_window}, "
                f"local_current_window={current_window}"
            )

            return True, checkpoint_sync_window

        except KeyError as e:
            tplr.logger.error(f"Invalid checkpoint format: missing key {e}")
            return False, 0
        except Exception as e:
            tplr.logger.error(f"Failed to load checkpoint: {e}")
            return False, 0

    async def post_peer_list(
        self,
        *,
        peers: list[int],
        reserve_peers: list[int] | None = None,
        first_effective_window: int,
        sync_window: int,
        initial_selection: bool,
    ):
        """Upload peer list and debug data as JSON to the node's R2 bucket.

        The first_effective_window is a future window (>current_window) from
        which this list peer list will be used.

        The following debugging fields are included in the JSON:
        - sync_window: when the peer list was updated in "validator time"
        - weights: weights for all UIDs, which were used to update the peer
          list (except for during the initial peer selection)
        - initial selection: whether this peer list is the first one in the
          current run
        """
        key = f"{PEERS_FILE_PREFIX}{first_effective_window}_v{__version__}.json"
        peers_and_weights = {
            "peers": peers,
            "reserve_peers": reserve_peers or [],
            "initial_selection": initial_selection,
            "sync_window": sync_window,
            "first_effective_window": first_effective_window,
        }

        # Create temporary JSON file
        temp_file = os.path.join(self.save_location, f"temp_{key}")
        try:
            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(peers_and_weights))

            await self.s3_put_object(key=key, file_path=temp_file)
            tplr.logger.info(f"PUT {key} <--")
        except Exception as e:
            tplr.logger.info(f"Failed to upload peer list: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # Start Window Operations
    async def post_start_window(self, start_window: int):
        """Upload the start window as a JSON object to the node's R2 bucket."""
        key = f"start_window_v{__version__}.json"
        start_window_data = {"start_window": start_window}

        # Create temporary JSON file
        temp_file = os.path.join(self.save_location, f"temp_{key}")
        try:
            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(start_window_data))

            await self.s3_put_object(key=key, file_path=temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    @async_s3_exception_catcher
    async def get_peer_list(
        self, fetch_previous: bool = False
    ) -> tuple[list[int], list[int], int] | None:
        tplr.logger.info(
            f"Looking for a {'previous' if fetch_previous else 'current'} peer list on a validator bucket"
        )
        while True:
            try:
                (
                    validator_bucket,
                    validator_uid,
                ) = await self._get_highest_stake_validator_bucket()

                if validator_bucket is None:
                    tplr.logger.warning(
                        "No highest staked validator bucket found. Retrying in 10 seconds."
                    )
                    await asyncio.sleep(10)
                    continue

                tplr.logger.info(
                    f"Attempting to fetch peer list from UID {validator_uid} bucket {validator_bucket.name}"
                )

                s3_client = await self._get_s3_client(validator_bucket)
                pattern = rf"^{PEERS_FILE_PREFIX}(?P<window>\d+)_v{re.escape(__version__)}\.json$"
                keys = []
                continuation_token = None

                while True:
                    list_args = {
                        "Bucket": validator_bucket.name,
                        "Prefix": PEERS_FILE_PREFIX,
                    }
                    if continuation_token:
                        list_args["ContinuationToken"] = continuation_token

                    response = await s3_client.list_objects_v2(**list_args)

                    for obj in response.get("Contents", []):
                        if re.match(pattern, obj["Key"]):
                            keys.append(obj["Key"])

                    if response.get("IsTruncated"):
                        continuation_token = response.get("NextContinuationToken")
                    else:
                        break

                if len(keys) == 0:
                    tplr.logger.info("No peer list files found")
                    return None

                # Parse windows from all keys
                window_to_key = {}
                for key in keys:
                    match = re.match(pattern, key)
                    if match:
                        window = int(match.group("window"))
                        window_to_key[window] = key

                if not window_to_key:
                    tplr.logger.error(
                        f"Failed to parse windows from peer list files. First "
                        f"{len(keys[:5])} peer list files are {keys[:5]}"
                    )
                    return None

                # Sort windows to find the most recent or the previous one
                window_to_keys = window_to_key.keys()

                if len(window_to_keys) == 0:
                    return None

                # If fetching previous, get the second most recent (if available)
                selected_window = None
                if fetch_previous and len(window_to_keys) > 1:
                    sorted_windows = sorted(window_to_keys, reverse=True)
                    selected_window = sorted_windows[1]  # Second most recent
                    tplr.logger.info(f"Selected previous window {selected_window}")
                elif fetch_previous and len(window_to_keys) <= 1:
                    tplr.logger.info(f"Found no previous window {selected_window}")
                    return None
                else:
                    selected_window = max(window_to_keys)  # Most recent
                    tplr.logger.info(f"Selected most recent window {selected_window}")

                selected_key = window_to_key[selected_window]

                peers_data = await self.s3_get_object(
                    key=selected_key, bucket=validator_bucket
                )

                if isinstance(peers_data, dict):
                    peers_dict = peers_data
                else:
                    peers_dict = json.loads(peers_data.decode("utf-8"))

                reserves = peers_dict.get("reserve_peers", [])
                return (
                    peers_dict["peers"],
                    reserves,
                    peers_dict["first_effective_window"],
                )

            except Exception as e:
                tplr.logger.error(f"Error fetching peer list: {e}")
                await asyncio.sleep(10)

    async def get_start_window(self, retries: int = -1) -> int | None:
        attempt = 0
        while retries == -1 or attempt < retries:
            try:
                (
                    validator_bucket,
                    validator_uid,
                ) = await self._get_highest_stake_validator_bucket()
                if validator_bucket is None:
                    tplr.logger.warning(
                        "No highest staked validator bucket found. Retrying in 10 seconds"
                    )
                    attempt += 1
                    await asyncio.sleep(10)
                    continue

                tplr.logger.info(
                    f"Attempting to fetch start_window from UID {validator_uid} bucket {validator_bucket.name}"
                )

                start_window_data = await self.s3_get_object(
                    key=f"start_window_v{__version__}.json", bucket=validator_bucket
                )

                if start_window_data is not None:
                    if isinstance(start_window_data, dict):
                        start_window_json = start_window_data
                    else:
                        start_window_json = json.loads(
                            start_window_data.decode("utf-8")
                        )

                    start_window = start_window_json["start_window"]
                    tplr.logger.info(f"Fetched start_window: {start_window}")
                    return start_window

                tplr.logger.warning(
                    "start_window.json not found or empty. Retrying in 10 seconds"
                )
                attempt += 1
                await asyncio.sleep(10)

            except Exception as e:
                tplr.logger.error(f"Error fetching start_window: {e}")
                attempt += 1
                await asyncio.sleep(10)

        tplr.logger.warning("Max retries exceeded while trying to fetch start_window")
        return None

    async def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        momentum,
        global_step,
        current_window,
        start_window,
    ):
        """Save checkpoint to R2 and local storage."""
        checkpoint_data = {
            "model_state_dict": {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": {
                k: v.cpu().clone() if torch.is_tensor(v) else v
                for k, v in optimizer.state_dict().items()
            },
            "scheduler_state_dict": scheduler.state_dict(),
            "momentum": {k: v.cpu().clone() for k, v in momentum.items()},
            "start_window": start_window,
            "current_window": current_window,
        }

        # save locally
        await self.put(
            state_dict=checkpoint_data,
            uid=str(self.uid),
            window=current_window,
            key="checkpoint",
            global_step=global_step,
            local=True,
        )

        # upload to R2
        await self.put(
            state_dict=checkpoint_data,
            uid=str(self.uid),
            window=current_window,
            key="checkpoint",
            global_step=global_step,
            local=False,
        )

        return True

    # async def _gather_window_batch(
    #     self,
    #     batch_windows: List[int],
    #     uid: str,
    #     peers: List[int],
    #     device: str,
    #     totalks: dict,
    #     global_step: int,
    # ) -> Dict[int, SimpleNamespace]:
    #     """Gather gradients for multiple windows in parallel."""
    #     try:
    #         gather_tasks = [
    #             self.gather(
    #                 my_uid=uid,
    #                 uids=peers,
    #                 window=w,
    #                 key="gradient",
    #                 timeout=30,
    #                 device=device,
    #                 totalks=totalks,
    #                 local=False,
    #                 stale_retention=100,
    #                 # Needs compressor!
    #             )
    #             for w in batch_windows
    #         ]
    #         # Wait for all gather tasks to complete
    #         batch_results = await asyncio.gather(*gather_tasks, return_exceptions=True)

    #         # Filter out exceptions and create window->result mapping
    #         result_dict = {w: None for w in batch_windows}  # Initialize with None
    #         for window, result in zip(batch_windows, batch_results):
    #             if not isinstance(result, Exception) and result is not None:
    #                 result_dict[window] = result

    #         return result_dict

    #     except Exception as e:
    #         tplr.logger.error(
    #             f"Failed to gather window batch {batch_windows}: {str(e)}"
    #         )
    #         return {
    #             w: None for w in batch_windows
    #         }  # Return dict with None values on failure

    def check_compressed_indices(
        self,
        param_name: str,
        idxs: torch.Tensor,
        totalk: int,
        allowed_topk: int | None = None,
        vals: torch.Tensor | None = None,
    ) -> None:
        allowed_topk = (
            min(self.hparams.topk_compression, totalk)
            if allowed_topk is None
            else min(allowed_topk, totalk)
        )

        def _bounds_check(t: torch.Tensor):
            """fast min/max bounds check"""
            if t.numel() == 0:
                raise ValueError(f"[{param_name}] empty index list")
            if t.min().item() < 0 or t.max().item() >= totalk:
                bad = t[(t < 0) | (t >= totalk)][0].item()
                raise ValueError(
                    f"[{param_name}] Index {bad} out of bounds (totalk = {totalk})"
                )

        # Handle 12-bit packed index format only
        if isinstance(idxs, torch.Tensor):
            if idxs.dtype != torch.uint8:
                raise ValueError(
                    f"[{param_name}] Expected uint8 for 12-bit packed indices, got {idxs.dtype}"
                )
            # 12-bit packed format is the only supported format
            if vals is None:
                raise ValueError(
                    f"[{param_name}] Values tensor required to validate 12-bit packed indices"
                )
            if idxs.numel() == 0:
                raise ValueError(f"[{param_name}] Empty packed indices tensor")

            # Unpack using the values shape
            try:
                unpacked = unpack_12bit_indices(idxs, vals.shape)
                # Validate that the last dimension matches allowed_topk
                if unpacked.shape[-1] != allowed_topk:
                    raise ValueError(
                        f"[{param_name}] Invalid topk dimension: "
                        f"shape[-1]={unpacked.shape[-1]} but expected {allowed_topk}"
                    )
                _bounds_check(unpacked)
            except Exception as e:
                raise ValueError(f"[{param_name}] Failed to unpack 12-bit indices: {e}")
        else:
            raise ValueError(f"[{param_name}] Expected tensor but got {type(idxs)}")

    @async_s3_exception_catcher
    async def get_debug_dict(self, window: int):
        """
        Get debug dictionary from validator bucket for a specific window.

        Args:
            window: Specific window to retrieve debug data for

        Returns:
            Debug dictionary or None if not found
        """
        (
            validator_bucket,
            validator_uid,
        ) = await self._get_highest_stake_validator_bucket()
        if not validator_bucket or validator_uid is None:
            tplr.logger.warning(
                "No validator bucket - cannot proceed with debug fetch"
            )
            return

        key = f"debug-{window}-{validator_uid}-v{tplr.__version__}.pt"
        tplr.logger.info(
            f"Attempting to retrieve debug dictionary for window {window} from validator {validator_uid}"
        )

        result = await self.s3_get_object(
            key=key,
            bucket=validator_bucket,
            timeout=20,
        )

        if result is None:
            tplr.logger.warning(f"No debug dictionary found for window {window}")
            return None

        tplr.logger.info(
            f"Successfully retrieved debug dictionary for window {window}"
        )
        return result

    def weighted_random_sample_no_replacement(
        self, candidates: list[int], weights: list[int], k: int
    ) -> list[int]:
        """
        Perform a weighted random sample (without replacement) of size k.
        candidates: list of items (uids).
        weights:    list of corresponding weights (integers or floats).
        k:          number of items to sample.
        Returns a list of selected items.
        """
        tplr.logger.debug("Starting weighted random sampling")
        tplr.logger.debug(f"Candidates: {candidates}")
        tplr.logger.debug(f"Weights: {weights}")
        tplr.logger.debug(f"Sample size (k): {k}")

        # Safety checks
        if not candidates or not weights or k <= 0:
            tplr.logger.warning("Invalid input detected. Returning empty list.")
            return []

        # Pair up each candidate with its weight
        pool = list(zip(candidates, weights))
        total_w = float(sum(weights))
        selected = []

        # If total weight is 0, return empty
        if total_w <= 0:
            tplr.logger.warning("Total weight is zero. Returning empty list")
            return []

        tplr.logger.debug(f"Initial total weight: {total_w}")

        for _ in range(min(k, len(candidates))):
            if total_w <= 0 or len(pool) == 0:
                tplr.logger.info("No more items to sample. Stopping early.")
                break

            # Draw a uniform sample in [0, total_w]
            r = random.uniform(0.0, total_w)
            tplr.logger.debug(f"Random threshold: {r}")
            cumulative = 0.0
            for idx, (uid, w) in enumerate(pool):
                cumulative += w
                if cumulative >= r:
                    # Found our pick
                    selected.append(uid)
                    tplr.logger.info(f"Selected item: {uid} with weight {w}")
                    # Remove from pool and subtract from total_w
                    total_w -= w
                    pool.pop(idx)
                    tplr.logger.debug(f"Updated total weight: {total_w}")
                    break

        tplr.logger.debug(f"Final selected items: {selected}")
        return selected
