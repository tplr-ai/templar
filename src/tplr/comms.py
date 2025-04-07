#!/usr/bin/env python3
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

import os
import json
import torch
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from types import SimpleNamespace

from aiobotocore.session import get_session

import tplr as tplr
from . import __version__
from .schemas import Bucket
from . import config

# Import component services that handle storage, chain syncing, peer management.
from tplr.peer_manager import PeerManager
from tplr.chain_sync import ChainSync
from tplr.catchup import CatchUpManager
from tplr.aggregation import AggregationManager
from tplr.debug_manager import DebugManager


class Comms:
    """
    Minimal communications module supporting expert-tagged gradients.
    In a real system, this would interact with remote storage (e.g. S3) and handle async transfers.
    For simulation, we use an in-memory dictionary.
    """
    def __init__(self, wallet, config=None, metagraph=None, hparams=None, uid=None):
        self.wallet = wallet
        self.config = config
        self.metagraph = metagraph
        self.hparams = hparams
        self.uid = uid
        self.storage = {}  # Simulate remote storage.

    async def gather(self, key: str, gradient: dict, **kwargs):
        """
        Simulate sending a gradient to the network by storing it under a key.
        """
        storage_key = f"{key}_{self.uid}"
        self.storage[storage_key] = gradient
        # Simulate network delay.
        await asyncio.sleep(0.05)
        print(f"Stored gradient for {key} by uid {self.uid}")
        return True

    async def get(self, key: str):
        """
        Simulate retrieving a gradient from a peer.
        Returns the first available gradient whose key matches and whose uid is not self.uid.
        """
        for k, v in self.storage.items():
            if key in k and str(self.uid) not in k:
                return v
        await asyncio.sleep(0.05)
        return None

    async def put(self, state_dict: dict, uid: str, window: int, key: str, global_step: int = 0, local: bool = True, stale_retention: int = 10) -> bool:
        """
        Simulate saving a checkpoint locally (or remotely).
        """
        filename = f"/tmp/{uid}_{key}_ws{window}_gs{global_step}.pt"
        torch.save(state_dict, filename)
        return True

    def start_background_tasks(self):
        """
        Start background tasks for active peer tracking and chain sync commitment fetching.
        """
        self.loop = asyncio.get_running_loop()
        # Start tracking active peers via peer_manager.
        self.loop.create_task(self.peer_manager.track_active_peers())
        # Start commitment fetching via the chain sync service.
        self.loop.create_task(self.chain.start_commitment_fetcher())

    async def gather(
        self,
        my_uid: str,
        uids: List[str],
        window: int,
        key: str,
        timeout: int,
        device: str,
        local: bool = True,
        stale_retention: int = 10,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
    ) -> Optional[SimpleNamespace]:
        """
        Gather gradients from a list of peer UIDs.
        """
        import time

        start_time = time.time()
        metrics = {"upload_bytes": 0, "download_bytes": 0}

        tplr.logger.debug(
            f"Starting gather for window {window} with time boundaries: {time_min} to {time_max}"
        )
        tplr.logger.debug(
            f"Gather params - my_uid: {my_uid}, key: {key}, timeout: {timeout}"
        )
        tplr.logger.debug(f"Target UIDs: {uids}")

        uid_state_dicts = {}
        valid_uids = []
        skipped_uids = []
        global_steps = {}

        async with self.client_semaphore:
            batch_tasks = [
                self.get_with_retry(
                    uid=uid,
                    window=window,
                    key=key,
                    local=local,
                    device=device,
                    timeout=timeout,
                    stale_retention=stale_retention,
                    time_min=time_min,
                    time_max=time_max,
                )
                for uid in uids
            ]
            batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for uid, response in zip(uids, batch_responses):
                if isinstance(response, Exception) or response is None:
                    tplr.logger.debug(f"Failure for {uid}: {response}")
                    skipped_uids.append(uid)
                    continue

                try:
                    state_dict_resp, global_step_resp = response
                except (TypeError, ValueError) as e:
                    tplr.logger.debug(f"Bad response format from {uid}: {e}")
                    skipped_uids.append(uid)
                    continue

                if not state_dict_resp:
                    tplr.logger.debug(f"Empty state_dict from {uid}")
                    skipped_uids.append(uid)
                    continue

                processed_state = {}
                for param_name, tensor in state_dict_resp.items():
                    if isinstance(tensor, torch.Tensor):
                        tensor = tensor.to(device)
                        if param_name.endswith("vals"):
                            norm = torch.norm(tensor)
                            normalized = tensor / (norm + 1e-8)
                            processed_state[param_name] = normalized
                        else:
                            processed_state[param_name] = tensor
                        metrics["download_bytes"] += (
                            tensor.element_size() * tensor.nelement()
                        )
                    else:
                        processed_state[param_name] = tensor

                uid_state_dicts[uid] = processed_state
                valid_uids.append(uid)
                global_steps[uid] = global_step_resp

        if not valid_uids:
            tplr.logger.info("No valid gradients received from any UID")
            return None

        total_time = time.time() - start_time
        tplr.logger.info(
            f"Gather completed in {total_time:.2f}s. Success rate: {len(valid_uids)}/{len(uids)}, "
            f"Upload: {metrics['upload_bytes']} bytes, Download: {metrics['download_bytes']} bytes"
        )
        result = SimpleNamespace(
            time=total_time,
            upload_bytes=metrics["upload_bytes"],
            download_bytes=metrics["download_bytes"],
            success_rate=len(valid_uids) / len(uids),
            state_dicts=uid_state_dicts,
            uids=valid_uids,
            global_steps=global_steps,
            skipped_uids=skipped_uids,
        )
        return result

    async def get_with_retry(
        self,
        uid,
        window,
        key,
        local,
        device,
        timeout,
        stale_retention,
        time_min,
        time_max,
    ):
        """
        Attempt to get data up to several times.
        """
        max_attempts = 3
        backoff_base = 1.5
        for attempt in range(max_attempts):
            try:
                return await self.get(
                    uid=uid,
                    window=window,
                    key=key,
                    local=local,
                    device=device,
                    timeout=timeout,
                    stale_retention=stale_retention,
                    time_min=time_min,
                    time_max=time_max,
                )
            except Exception as e:
                if attempt < max_attempts - 1:
                    wait_time = backoff_base**attempt
                    tplr.logger.debug(
                        f"Retrying UID {uid} in {wait_time:.1f}s due to error: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    tplr.logger.warning(
                        f"Get failed for UID {uid} after {max_attempts} attempts: {e}"
                    )
                    raise Exception(
                        f"Get failed for UID {uid} after {max_attempts} attempts: {e}"
                    )

    def _get_own_bucket(self, bucket_type: str, rw: str) -> Bucket:
        """
        Look up bucket configuration from the config and return a Bucket instance.
        """
        bucket_conf = config.BUCKET_SECRETS.get(bucket_type)
        tplr.logger.debug("Bucket config for %s: %s", bucket_type, bucket_conf)
        if not bucket_conf:
            raise ValueError(f"No bucket configuration found for '{bucket_type}'.")

        name = bucket_conf.get("name", "").strip()
        if not name:
            raise ValueError(f"Bucket name for '{bucket_type}' must not be empty.")

        account_id = bucket_conf.get("account_id", "").strip()
        if not account_id:
            raise ValueError(
                f"Bucket account_id for '{bucket_type}' must not be empty."
            )

        creds = bucket_conf.get("credentials", {}).get(rw, {})
        access_key = creds.get("access_key_id", "").strip()
        secret_key = creds.get("secret_access_key", "").strip()
        if not access_key or not secret_key:
            raise ValueError(
                f"Bucket credentials for '{bucket_type}' in '{rw}' mode must not be empty."
            )

        bucket = Bucket(
            name=name,
            account_id=account_id,
            access_key_id=access_key,
            secret_access_key=secret_key,
        )
        tplr.logger.debug("Created Bucket: %s", bucket)
        return bucket

    async def post_start_window(self, start_window: int):
        """
        Post start_window as JSON data to the bucket.
        """
        data = json.dumps({"start_window": start_window}).encode("utf-8")
        key = f"start_window_v{__version__}.json"
        return await self.storage.store_bytes(data=data, key=key, bucket=self.bucket)

    async def get_start_window(self) -> int:
        """
        Retrieve start_window from the highest staked validator's bucket.
        """
        while True:
            try:
                # Use chain to fetch the highest staked validator bucket.
                (
                    validator_bucket,
                    validator_uid,
                ) = await self.chain._get_highest_stake_validator_bucket()
                if validator_bucket is None:
                    tplr.logger.warning(
                        "No highest staked validator bucket found. Retrying in 10 seconds."
                    )
                    await asyncio.sleep(10)
                    continue

                tplr.logger.info(f"Fetching start_window from UID {validator_uid}")
                key = f"start_window_v{__version__}.json"
                start_window_data = await self.storage.get_bytes(
                    key=key, bucket=validator_bucket
                )
                if start_window_data:
                    if isinstance(start_window_data, (bytes, bytearray)):
                        start_window_json = json.loads(
                            start_window_data.decode("utf-8")
                        )
                    elif isinstance(start_window_data, dict):
                        start_window_json = start_window_data
                    else:
                        start_window_json = {"start_window": 0}
                    start_window = start_window_json["start_window"]
                    tplr.logger.info(f"Fetched start_window: {start_window}")
                    return start_window

                tplr.logger.warning(
                    "start_window.json not found or empty. Retrying in 10 seconds."
                )
                await asyncio.sleep(10)
            except Exception as e:
                tplr.logger.error(f"Error fetching start_window: {e}")
                await asyncio.sleep(10)

    async def load_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        device: str,
    ) -> Tuple[
        bool, dict, int, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler
    ]:
        """
        Load the latest available checkpoint.
        Returns a tuple: (success, momentum, global_step, optimizer, scheduler)
        """
        bucket = self.chain.get_own_bucket("gradients", "read")
        if bucket is None:
            tplr.logger.warning("No bucket available for loading checkpoint")
            return False, {}, 0, optimizer, scheduler

        latest_window = None
        latest_checkpoint = None

        # List available checkpoints using storage helper.
        try:
            import boto3
            from botocore.client import Config
            import re

            s3_config = Config(
                region_name="auto", signature_version="s3v4", max_pool_connections=256
            )
            s3_client = boto3.client(
                "s3",
                endpoint_url=f"https://{bucket.account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=bucket.access_key_id,
                aws_secret_access_key=bucket.secret_access_key,
                config=s3_config,
            )

            response = s3_client.list_objects_v2(
                Bucket=bucket.name, Prefix="checkpoint-"
            )
            current_version = __version__
            tplr.logger.info(f"Looking for checkpoints with version {current_version}")

            pattern = re.compile(r"^checkpoint-(\d+)-(\d+)-v(.+)\.pt$")
            if "Contents" in response:
                for obj in response["Contents"]:
                    key = obj["Key"]
                    m = pattern.match(key)
                    if not m:
                        tplr.logger.debug(f"Skipping invalid checkpoint format: {key}")
                        continue
                    try:
                        checkpoint_window = int(m.group(1))
                        checkpoint_uid = int(m.group(2))
                        checkpoint_version = m.group(3)
                    except Exception as e:
                        tplr.logger.debug(f"Error parsing key {key}: {e}")
                        continue
                    if checkpoint_version != current_version:
                        tplr.logger.debug(
                            f"Skipping checkpoint {key} due to version mismatch"
                        )
                        continue
                    # Only consider checkpoints from the highest staked validator.
                    (
                        _,
                        highest_validator_uid,
                    ) = await self.chain._get_highest_stake_validator_bucket(
                        refresh_commitments=True
                    )
                    if (
                        highest_validator_uid is None
                        or checkpoint_uid != highest_validator_uid
                    ):
                        tplr.logger.debug(
                            f"Skipping checkpoint {key} (not from highest staked validator)"
                        )
                        continue
                    if latest_window is None or checkpoint_window > latest_window:
                        latest_window = checkpoint_window
                        latest_checkpoint = key

            if latest_checkpoint:
                tplr.logger.info(
                    f"Found checkpoint: {latest_checkpoint} (window {latest_window})"
                )
            else:
                tplr.logger.info("No valid checkpoints found; starting from scratch.")
                return False, {}, 0, optimizer, scheduler

        except Exception as e:
            tplr.logger.warning(f"Error listing checkpoint files: {e}")
            return False, {}, 0, optimizer, scheduler

        try:
            temp_path = os.path.join(self.storage.temp_dir, latest_checkpoint)
            tplr.logger.info(
                f"Downloading checkpoint {latest_checkpoint} to {temp_path}"
            )
            result = await self.storage.s3_get_object(
                key=latest_checkpoint, bucket=bucket, file_path=temp_path, timeout=60
            )
            if not result:
                tplr.logger.warning(
                    f"Failed to download checkpoint {latest_checkpoint}"
                )
                return False, {}, 0, optimizer, scheduler

            tplr.logger.info(f"Loading checkpoint from {temp_path}")
            checkpoint = await asyncio.to_thread(
                torch.load, temp_path, map_location=device
            )
            if "model_state_dict" not in checkpoint:
                tplr.logger.error("Checkpoint missing model_state_dict key")
                return False, {}, 0, optimizer, scheduler

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            momentum = checkpoint["momentum"]
            global_step = checkpoint["global_step"]

            tplr.logger.info(f"Loaded checkpoint with global_step={global_step}")
            os.remove(temp_path)

            return True, momentum, global_step, optimizer, scheduler

        except Exception as e:
            tplr.logger.error(f"Error loading checkpoint data: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False, {}, 0, optimizer, scheduler

    async def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        momentum,
        global_step,
        current_window,
        start_window,
    ) -> bool:
        """
        Save checkpoint data both locally and remotely.
        """
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

        # Save locally.
        local_ok = await self.put(
            state_dict=checkpoint_data,
            uid=str(self.uid),
            window=current_window,
            key="checkpoint",
            global_step=global_step,
            local=True,
        )
        # Then upload to remote storage.
        remote_ok = await self.put(
            state_dict=checkpoint_data,
            uid=str(self.uid),
            window=current_window,
            key="checkpoint",
            global_step=global_step,
            local=False,
        )

        return local_ok and remote_ok

    # Internal helper methods used by CatchUpManager and AggregationManager:
    async def _gather_window_batch(
        self,
        batch_windows: list,
        uid: str,
        peers: list,
        device: str,
        totalks: dict,
        global_step: int,
    ) -> dict:
        try:
            gather_tasks = [
                self.gather(
                    my_uid=uid,
                    uids=peers,
                    window=w,
                    key="gradient",
                    timeout=30,
                    device=device,
                    local=False,
                    stale_retention=100,
                )
                for w in batch_windows
            ]
            batch_results = await asyncio.gather(*gather_tasks, return_exceptions=True)
            result_dict = {w: None for w in batch_windows}
            for window, result in zip(batch_windows, batch_results):
                if not isinstance(result, Exception) and result is not None:
                    result_dict[window] = result
            return result_dict
        except Exception as e:
            tplr.logger.error(
                f"Failed to gather window batch {batch_windows}: {str(e)}"
            )
            return {w: None for w in batch_windows}

    async def _apply_gathered_gradients(
        self,
        gather_result,
        model,
        optimizer,
        scheduler,
        transformer,
        compressor,
        device: str,
        window: int,
        global_step: int,
    ):
        try:
            if not gather_result or not gather_result.state_dict:
                return False, global_step
            model.train()
            optimizer.zero_grad()
            model.zero_grad()
            for n, p in model.named_parameters():
                idxs = getattr(gather_result.state_dict, f"{n}idxs", None)
                vals = getattr(gather_result.state_dict, f"{n}vals", None)
                if idxs is not None and vals is not None:
                    if not isinstance(idxs, (list, tuple)):
                        idxs = [idxs]
                    if not isinstance(vals, (list, tuple)):
                        vals = [vals]
                    new_grad = transformer.decode(
                        compressor.batch_decompress(
                            p.to(device),
                            idxs,
                            vals,
                            transformer.shapes[n],
                            transformer.totalks[n],
                        )
                    )
                    if p.grad is None:
                        p.grad = new_grad
                    else:
                        p.grad.copy_(new_grad)
                    p.grad.sign_()
            optimizer.step()
            scheduler.step()
            global_step += 1
            tplr.logger.info(
                f"Applied gradients for window {window}, global_step => {global_step}"
            )
            return True, global_step
        except Exception as e:
            tplr.logger.error(
                f"Failed to apply gradients for window {window}: {str(e)}"
            )
            return False, global_step


# TODO:
#   - Evaluate whether additional catch-up or aggregation helper functions
#     should be moved to separate modules.
#   - Confirm that chain sync (_get_highest_stake_validator_bucket) provides all information.
