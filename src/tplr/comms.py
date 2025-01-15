# src/tplr/comms.py

import os
import re
import time
import torch
import asyncio
import aiofiles
import tempfile
import bittensor as bt
from typing import List, Dict, Optional
from types import SimpleNamespace

# Import huggingface_hub for file upload/download
from huggingface_hub import upload_file, hf_hub_download, HfApi

import tplr
from tplr import __version__
from tplr.config import BUCKET_SECRETS  # If you still want chain-based “commitment” data
from tplr.chain import ChainManager
from tplr.schemas import Bucket

from tqdm.asyncio import tqdm
import numpy as np
import psutil
import mmap


LOCAL_TMP_DIR = "/tmp/local_store"


class Comms(ChainManager):
    """
    This class was originally S3-based. Now, we switch to using the huggingface_hub library
    for uploading/downloading artifact files. The rest of the chain-based logic, local file
    handling, and “active peers” remain.
    """

    def __init__(
        self,
        wallet: "bt.wallet",
        save_location: str = "/tmp",
        key_prefix: str = "model",
        config=None,
        netuid=None,
        metagraph=None,
        hparams=None,
        uid=None,
        **kwargs,
    ):
        # Bittensor wallet, chain, etc.
        self.wallet = wallet
        self.uid = uid
        self.temp_dir = os.path.join("/tmp", f"templar_{self.uid}")
        os.makedirs(self.temp_dir, exist_ok=True)

        # If you stored old R2 credentials in BUCKET_SECRETS, you can repurpose 
        # them or store new HF config. For demonstration, we'll keep it around:
        self.bucket = self.get_own_bucket()  # now might be “mocked” or unused

        super().__init__(
            config=config,
            netuid=netuid,
            metagraph=metagraph,
            hparams=hparams,
            wallet=self.wallet,
            bucket=self.bucket,
        )

        # Instead of S3, we’ll keep track of HF repo info.
        # You might store these in environment variables or on-chain:
        self.huggingface_token = os.environ.get("HUGGINGFACE_TOKEN", "")
        self.huggingface_repo_id = os.environ.get("HUGGINGFACE_REPO", "my-user/templar-checkpoints")

        # Local directory for ephemeral caching
        hotkey = self.wallet.hotkey.ss58_address
        self.save_location = os.path.join("/tmp", f"hotkey_{hotkey}")
        os.makedirs(self.save_location, exist_ok=True)
        self.key_prefix = key_prefix

        # We keep a generic lock and set of background tasks
        self.lock = asyncio.Lock()
        self.active_peers = set()  
        self.active_check_interval = self.hparams.active_check_interval
        self.recent_windows = self.hparams.recent_windows

    def get_own_bucket(self) -> Bucket:
        """
        If you still want to parse a chain-based "commitment" for the HF token or repo,
        do so here. Otherwise, just return a dummy bucket or remove this method entirely.
        """
        return Bucket(
            name="HF_REPO_UNUSED",
            account_id="HF_ACCOUNT_UNUSED",
            access_key_id="dummy_access_key",
            secret_access_key="dummy_secret_key",
        )

    def start_background_tasks(self):
        """Attach background tasks like track_active_peers if you still want them."""
        self.loop = asyncio.get_running_loop()
        self.loop.create_task(self.track_active_peers())

    # ------------------------------------------------------------------------
    # Local File Management (unchanged)
    # ------------------------------------------------------------------------
    def delete_local_directory(self, path: str):
        if not os.path.exists(path):
            return
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(path)

    async def cleanup_local_data(self, uid: str, current_window: int, stale_retention: int):
        """Clean up stale local data for a given uid (unchanged)."""
        user_dir = os.path.join(LOCAL_TMP_DIR, str(uid))
        if not os.path.exists(user_dir):
            return

        min_allowed_window = current_window - stale_retention
        for wdir in os.listdir(user_dir):
            if wdir.isdigit():
                w = int(wdir)
                if w < min_allowed_window:
                    old_path = os.path.join(user_dir, wdir)
                    tplr.logger.debug(f"Removing stale local directory: {old_path}")
                    try:
                        self.delete_local_directory(old_path)
                    except Exception as e:
                        tplr.logger.debug(f"Error removing stale directory {old_path}: {e}")

    # ------------------------------------------------------------------------
    # Hugging Face Upload / Download
    # ------------------------------------------------------------------------
    async def hf_upload_file(self, local_path: str, target_path: str):
        """
        Use huggingface_hub.upload_file to push a local file to the repository.
        """
        try:
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=target_path,
                repo_id=self.huggingface_repo_id,
                token=self.huggingface_token,
                repo_type="model",  # or 'dataset' if you prefer
            )
            tplr.logger.debug(f"Uploaded {local_path} to HF repo {target_path}.")
        except Exception as e:
            tplr.logger.error(f"HF Upload error for {local_path}: {e}")
            raise

    async def hf_download_file(self, target_path: str, local_path: str):
        """
        Use huggingface_hub.hf_hub_download to pull a file from the repository.
        """
        try:
            downloaded = hf_hub_download(
                repo_id=self.huggingface_repo_id,
                filename=target_path,
                repo_type="model",
                token=self.huggingface_token,
            )
            # Move or copy from downloaded to local_path
            os.replace(downloaded, local_path)
            tplr.logger.debug(f"Downloaded {target_path} from HF to {local_path}.")
            return True
        except Exception as e:
            tplr.logger.debug(f"HF Download error for {target_path}: {e}")
            return False

    async def cleanup_hf_data(self, uid: str, current_window: int, stale_retention: int):
        """
        If you want to remove older artifacts from the HF repo, implement some listing
        logic (e.g. HfApi().list_repo_files()) and delete older files. We skip this for now.
        """
        pass

    # ------------------------------------------------------------------------
    # PUT / GET methods for gradient or checkpoint files
    # ------------------------------------------------------------------------
    async def put(
        self,
        state_dict: dict,
        uid: str,
        window: int,
        key: str,
        global_step: int = 0,
        local: bool = True,
        stale_retention: int = 10,
    ):
        """
        Save `state_dict` either locally or on the Hugging Face Hub. 
        Replaces old R2 logic with HF calls.
        """
        tplr.logger.debug(f"PUT {uid}/{window}/{key} -->")

        # Construct a filename (similar to the old R2 naming)
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        temp_dir = os.path.join("/tmp", str(self.uid))
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"temp_{filename}")

        try:
            # Prepare data: if 'checkpoint', store full checkpoint; else store {state_dict, global_step}
            if key == "checkpoint":
                save_data = state_dict
            else:
                save_data = {
                    "state_dict": state_dict,
                    "global_step": global_step,
                }
            # Save to local temp
            torch.save(save_data, temp_file_path)

            if local:
                # Keep local (unchanged from R2 code):
                await self.cleanup_local_data(uid=uid, current_window=window, stale_retention=stale_retention)
                local_dir = os.path.join(LOCAL_TMP_DIR, str(uid), str(window))
                os.makedirs(local_dir, exist_ok=True)
                final_path = os.path.join(local_dir, filename)
                os.replace(temp_file_path, final_path)
            else:
                # Instead of S3, call Hugging Face upload
                await self.cleanup_hf_data(uid=uid, current_window=window, stale_retention=stale_retention)
                target_path = f"{uid}/{window}/{filename}"  # path in HF repo
                await self.hf_upload_file(temp_file_path, target_path)

        except Exception as e:
            tplr.logger.debug(f"PUT error {uid}/{window}/{key}: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        tplr.logger.debug(f"PUT {uid}/{window}/{key} <--")

    async def get(
        self,
        uid: str,
        window: int,
        key: str,
        timeout: int = 5,
        local: bool = True,
        stale_retention: int = 10,
    ) -> Optional[dict]:
        """
        Retrieve a gradient or checkpoint file. 
        Local or from Hugging Face if `local=False`.
        """
        filename = f"{key}-{window}-{uid}-v{__version__}.pt"
        tplr.logger.debug(f"GET {filename} -->")

        try:
            # Temporary path
            temp_dir = os.path.join("/tmp", str(self.uid))
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, f"temp_{filename}")

            if local:
                await self.cleanup_local_data(uid, window, stale_retention)
                local_path = os.path.join(LOCAL_TMP_DIR, str(uid), str(window), filename)
                if not os.path.exists(local_path):
                    tplr.logger.debug(f"Local file not found: {local_path}")
                    return None

                loaded_data = torch.load(local_path, weights_only=True)
                if key == "checkpoint":
                    return loaded_data, None
                return loaded_data.get("state_dict"), loaded_data.get("global_step", 0)

            else:
                # Use HF
                target_path = f"{uid}/{window}/{filename}"
                success = await self.hf_download_file(target_path, temp_file_path)
                if not success:
                    return None

                loaded_data = torch.load(temp_file_path, weights_only=True)
                if key == "checkpoint":
                    return loaded_data, None
                return loaded_data.get("state_dict"), loaded_data.get("global_step", 0)

        except Exception as e:
            tplr.logger.debug(f"GET error {filename}: {e}")
            return None
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
    ) -> Optional[dict]:
        """
        GET with a time-based retry. 
        If the file is not found, it attempts again until `timeout`.
        """
        start_time = time.time()
        end_time = start_time + timeout

        while True:
            if time.time() >= end_time:
                tplr.logger.debug(f"GET {uid}/{window}/{key} timed out.")
                return None

            result = await self.get(
                uid=uid,
                window=window,
                key=key,
                local=local,
                stale_retention=stale_retention,
            )
            if result is not None:
                return result

            await asyncio.sleep(0.1)

    async def gather(
        self,
        state_dict: Optional[Dict[str, torch.Tensor]],
        my_uid: str,
        uids: List[str],
        window: int,
        key: str,
        timeout: int,
        device: str,
        global_step: int,
        local: bool = True,
        stale_retention: int = 10,
    ) -> Optional[SimpleNamespace]:
        """
        Orchestrates the process of:
          - Uploading our local state (if provided).
          - Downloading others' states.
          - Aggregating them into a single structure.
        """
        start_time = time.time()
        metrics = {"upload_bytes": 0, "download_bytes": 0, "successes": []}

        # Put own state
        if my_uid is not None and state_dict is not None:
            await self.put(
                state_dict=state_dict,
                uid=str(my_uid),
                window=window,
                key=key,
                global_step=global_step,
                local=local,
                stale_retention=stale_retention,
            )
            # Approx. upload size
            metrics["upload_bytes"] += sum(t.nelement() * t.element_size() for t in state_dict.values())

        await asyncio.sleep(0.1)

        # Gather from peers
        gather_tasks = [
            self.get_with_retry(
                uid=uid,
                window=window,
                key=key,
                timeout=timeout,
                local=local,
                stale_retention=stale_retention,
            )
            for uid in uids
        ]

        aggregated_state_dict = {}
        valid_uids = []
        global_steps = []

        responses = await asyncio.gather(*gather_tasks)
        for idx, response in enumerate(responses):
            uid_i = uids[idx]
            if response is None:
                tplr.logger.debug(f"No data from UID {uid_i}")
                continue

            try:
                state_dict_resp, global_step_resp = response
            except (TypeError, ValueError):
                # Possibly not a 2-tuple
                tplr.logger.debug(f"Invalid response format from UID {uid_i}")
                continue

            if state_dict_resp is None:
                tplr.logger.debug(f"Empty state dict from UID {uid_i}")
                continue

            valid_uids.append(uid_i)
            global_steps.append(global_step_resp)

            for param_name, tensor in state_dict_resp.items():
                if param_name not in aggregated_state_dict:
                    aggregated_state_dict[param_name] = []
                aggregated_state_dict[param_name].append(tensor.to(device))
                metrics["download_bytes"] += tensor.nelement() * tensor.element_size()

        if not valid_uids:
            tplr.logger.info("No valid gradients received from any UID")
            return None

        result = SimpleNamespace(
            time=time.time() - start_time,
            upload_bytes=metrics["upload_bytes"],
            download_bytes=metrics["download_bytes"],
            success_rate=len(valid_uids) / len(uids),
            state_dict=SimpleNamespace(**aggregated_state_dict),
            uids=valid_uids,
            global_steps=global_steps,
        )
        tplr.logger.debug(f"Gathered from UIDs: {valid_uids}")
        return result

    # ------------------------------------------------------------------------
    # Active Miner Checking
    # ------------------------------------------------------------------------
    async def is_miner_active(self, uid: int, recent_windows: int = 3) -> bool:
        """
        Check if a miner has uploaded gradient files in the last `recent_windows`.
        We can do a HEAD check in Hugging Face or see if it’s found locally.

        For demonstration, we do a naive approach with `hf_hub_download`:
        If any gradient file in [current_window - recent_windows, ..., current_window]
        exists, we consider them active.
        """
        tplr.logger.debug(f"Checking if UID {uid} is active")

        if not hasattr(self, "current_window") or self.current_window is None:
            tplr.logger.error("current_window is not set in comms. Please set comms.current_window.")
            return False

        hf_api = HfApi(token=self.huggingface_token)
        current_window = self.current_window
        for w in range(current_window - recent_windows, current_window + 1):
            target_path = f"{uid}/{w}/gradient-{w}-{uid}-v{__version__}.pt"
            # See if file is in the HF repo
            try:
                files = hf_api.list_repo_files(repo_id=self.huggingface_repo_id, repo_type="model")
                if target_path in files:
                    tplr.logger.debug(f"Found gradient for UID {uid} window {w}")
                    return True
            except Exception as e:
                tplr.logger.error(f"Error checking HF repo for UID {uid}, window {w}: {e}")
                return False

        return False

    async def track_active_peers(self):
        while True:
            active_peers = set()
            tasks = []
            semaphore = asyncio.Semaphore(10)

            tplr.logger.debug(f"Commitments: {self.commitments}")

            async def check_peer(uid):
                async with semaphore:
                    is_active = await self.is_miner_active(uid, recent_windows=self.recent_windows)
                    if is_active:
                        active_peers.add(uid)

            for uid in self.commitments.keys():
                tasks.append(check_peer(uid))

            await asyncio.gather(*tasks)
            self.active_peers = active_peers
            tplr.logger.info(f"Updated active peers: {[int(u) for u in self.active_peers]}")
            await asyncio.sleep(self.active_check_interval)

    async def get_latest_checkpoint(self):
        """
        Example placeholder: If you want to find the “highest stake validator’s
        checkpoint” on HF. This logic is up to you. We do a naive approach:
        1. Get the highest stake UID
        2. Look for the latest checkpoint file in their subfolder
        """
        try:
            validator_uid = self.metagraph.S.argmax().item()
            tplr.logger.info(f"Found validator with highest stake: {validator_uid}")

            if validator_uid is None:
                tplr.logger.info("No active validators found")
                return None

            # This is naive: we might list all checkpoint-* files in that UID’s subfolders.
            # For example, you can parse them by last modified or a naming scheme. 
            hf_api = HfApi(token=self.huggingface_token)
            files = hf_api.list_repo_files(repo_id=self.huggingface_repo_id, repo_type="model")

            # Filter for “checkpoint-*” files under folder = f"{validator_uid}/..."
            pattern = re.compile(rf"^{validator_uid}/(\d+)/checkpoint-\1-{validator_uid}-v.*\.pt$")
            checkpoint_files = []
            for f in files:
                match = pattern.match(f)
                if match:
                    # parse out the window from match.group(1)
                    w = int(match.group(1))
                    checkpoint_files.append((f, w))

            if not checkpoint_files:
                tplr.logger.info("No checkpoint files found")
                return None

            # Sort by window descending
            checkpoint_files.sort(key=lambda x: x[1], reverse=True)
            latest = checkpoint_files[0]
            latest_key = latest[0]
            latest_window = latest[1]

            tplr.logger.info(f"Latest checkpoint: {latest_key} from window {latest_window}")

            # Download and return 
            # Our “get” method expects (checkpoint_data, None) for a checkpoint
            checkpoint_data, _ = await self.get(str(validator_uid), latest_window, "checkpoint", local=False)
            return checkpoint_data, latest_window

        except Exception as e:
            tplr.logger.error(f"Error getting latest checkpoint from HF: {e}")
            return None
