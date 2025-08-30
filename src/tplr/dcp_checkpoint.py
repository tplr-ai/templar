# Copyright (c) 2025 tplr.ai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import asyncio
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    ValueType,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.checkpoint.state_dict_loader import load
from torch.distributed.checkpoint.state_dict_saver import async_save
from torch.distributed.checkpoint.stateful import Stateful


# ── Model-only Stateful (Titan-compatible distributed state dicts) ─────────────
class AppState(Stateful):
    def __init__(self, model):
        self.model = model

    def state_dict(self) -> dict[str, ValueType]:
        return get_model_state_dict(self.model)

    def load_state_dict(self, state: dict[str, ValueType]) -> None:
        set_model_state_dict(self.model, state)


# ── Utils ─────────────────────────────────────────────────────────────────────
def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _owner_for(filename: str, world: int) -> int:
    h = hashlib.blake2b(filename.encode(), digest_size=4).digest()
    return int.from_bytes(h, "little") % max(world, 1)


def _is_meta(p: Path) -> bool:
    n = p.name.lower()
    return n == ".metadata" or n.endswith(".metadata") or n.endswith(".json")


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


@dataclass(slots=True)
class Layout:
    version: str
    window: int | None = None

    @property
    def prefix(self) -> str:
        base = f"checkpoints/{self.version}"
        return f"{base}/{self.window}" if self.window is not None else base


class DCPCheckpointer:
    """
    TorchTitan‑friendly DCP checkpoint manager.

    • All ranks in the saving PG call save_local()  → sharded checkpoint to repo_root/checkpoints/…
    • Per‑rank upload split: each rank uploads its shard subset; rank‑0 uploads metadata + _LATEST.json
    • Optional background upload; delete local after success
    • Download + DCP load (reshards to miner topology)
    """

    def __init__(self, comms, *, uid: int, version: str, repo_root: str | Path = "."):
        self.comms = comms
        self.uid = int(uid)
        self.version = version
        self.repo_root = Path(repo_root).resolve()
        self._bg: set[asyncio.Task] = set()

    # ── Paths ──────────────────────────────────────────────────────────────────
    def _local_dir(self, layout: Layout) -> Path:
        d = self.repo_root / layout.prefix
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Save (async local first; Titan-style if available) ─────────────────────
    async def save_local(
        self,
        *,
        model,
        window: int,
        sync_window: int,
        topology: str = "FSDP",
        process_group: dist.ProcessGroup | None = None,  # HSDP: pass shard PG
    ) -> Path:
        """
        Call on ALL ranks in the saving PG (global PG for FSDP/TP/DP; one shard_group PG for HSDP).
        Produces sharded files + .metadata under repo_root/checkpoints/<version>/<window>.
        """
        layout = Layout(self.version, window)
        out_dir = self._local_dir(layout)
        state = {"app": AppState(model)}

        # Prefer DCP async_save (TorchTitan’s path); fallback to thread if absent.
        fut = async_save(
            state_dict=state,
            checkpoint_id=str(out_dir),
            process_group=process_group,
        )
        fut.result()  # ensure the local files exist before upload

        if _rank() == 0:
            sidecar = {
                "version": self.version,
                "window": int(window),
                "sync_window": int(sync_window),
                "world_size_at_save": int(_world()),
                "topology": topology,
                "uid": self.uid,
            }
            (out_dir / "extra_metadata.json").write_text(json.dumps(sidecar, indent=2))

        _barrier()
        return out_dir

    # ── Upload (per‑rank; rank‑0 handles metadata and pointer) ─────────────────
    async def upload(
        self,
        *,
        window: int,
        background: bool = False,
        delete_local_on_success: bool = True,
    ) -> None:
        layout = Layout(self.version, window)
        local_dir = self._local_dir(layout)

        async def _do() -> None:
            world, r = _world(), _rank()
            files = [p for p in local_dir.iterdir() if p.is_file()]

            tasks: list[asyncio.Task] = []
            # rank‑0: metadata + version pointer
            if r == 0:
                for p in (q for q in files if _is_meta(q)):
                    tasks.append(
                        asyncio.create_task(
                            self.comms.s3_put_object(
                                key=f"{layout.prefix}/{p.name}", file_path=str(p)
                            )
                        )
                    )
                tmp = local_dir / "_LATEST.json"
                tmp.write_text(json.dumps({"latest_window": int(window)}, indent=2))
                tasks.append(
                    asyncio.create_task(
                        self.comms.s3_put_object(
                            key=f"checkpoints/{self.version}/_LATEST.json",
                            file_path=str(tmp),
                        )
                    )
                )

            # all ranks: shard/data files by deterministic ownership
            for p in (q for q in files if not _is_meta(q)):
                if _owner_for(p.name, world) == r:
                    tasks.append(
                        asyncio.create_task(
                            self.comms.s3_put_object(
                                key=f"{layout.prefix}/{p.name}", file_path=str(p)
                            )
                        )
                    )

            if tasks:
                await asyncio.gather(*tasks)
            _barrier()

            # delete local after everyone finished and uploads succeeded
            if delete_local_on_success and r == 0:
                try:
                    shutil.rmtree(local_dir, ignore_errors=True)
                except Exception:
                    pass

        if background:
            t = asyncio.create_task(_do())
            self._bg.add(t)
            t.add_done_callback(self._bg.discard)
        else:
            await _do()

    async def flush_background_uploads(self) -> None:
        if self._bg:
            await asyncio.gather(*list(self._bg))
            self._bg.clear()

    # ── Bucket selection (prefer highest-staked validator, fallback to own) ───
    async def _choose_read_bucket(self, prefer_highest_staked: bool = True):
        """
        Returns the bucket to read from. If prefer_highest_staked is True,
        try the highest-staked validator's bucket first; fallback to own bucket.
        """
        if prefer_highest_staked:
            try:
                bucket, _uid = await self.comms._get_highest_stake_validator_bucket()
                if bucket is not None:
                    return bucket
            except Exception:
                pass
        return self.comms.bucket

    # ── Discover latest remote window ──────────────────────────────────────────
    async def _discover_latest(
        self, *, prefer_highest_staked: bool = True
    ) -> int | None:
        """
        Find the newest window, trying highest-staked validator bucket first, then own.
        """
        # Attempt in order: highest-staked → own
        bucket = await self._choose_read_bucket(
            prefer_highest_staked=prefer_highest_staked
        )
        s3 = await self.comms._get_s3_client(bucket)
        # pointer first
        try:
            obj = await self.comms.s3_get_object(
                key=f"checkpoints/{self.version}/_LATEST.json", bucket=bucket
            )
            if isinstance(obj, dict) and "latest_window" in obj:
                return int(obj["latest_window"])
        except Exception:
            pass
        # list fallback
        version_prefix = Layout(self.version).prefix + "/"  # ensure trailing slash
        windows: set[int] = set()
        cont = None
        while True:
            args = {"Bucket": bucket.name, "Prefix": version_prefix}
            if cont:
                args["ContinuationToken"] = cont
            resp = await s3.list_objects_v2(**args)
            for o in resp.get("Contents", []) or []:
                key = o.get("Key", "")
                m = re.match(rf"^{re.escape(version_prefix)}(\d+)/", key)
                if m:
                    windows.add(int(m.group(1)))
            if not resp.get("IsTruncated"):
                break
            cont = resp.get("NextContinuationToken")
        if windows:
            return max(windows)
        return None

    # ── Download: safe default (each rank mirrors full folder) ─────────────────
    async def download_all(
        self, *, window: int | None = None, prefer_highest_staked: bool = True
    ) -> Path | None:
        if window is None:
            window = await self._discover_latest(
                prefer_highest_staked=prefer_highest_staked
            )
            if window is None:
                return None
        layout = Layout(self.version, window)
        local_dir = self._local_dir(layout)

        bucket = await self._choose_read_bucket(
            prefer_highest_staked=prefer_highest_staked
        )
        s3 = await self.comms._get_s3_client(bucket)
        got_any = False
        cont = None
        while True:
            args = {"Bucket": bucket.name, "Prefix": f"{layout.prefix}/"}
            if cont:
                args["ContinuationToken"] = cont
            resp = await s3.list_objects_v2(**args)
            for obj in resp.get("Contents", []) or []:
                key = obj.get("Key")
                if key:
                    got_any = True
                    await self.comms.s3_get_object(
                        key=key, bucket=bucket, load_data=False, show_progress=False
                    )
            if not resp.get("IsTruncated"):
                break
            cont = resp.get("NextContinuationToken")
        if got_any:
            return local_dir
        return None

    # ── Download: shared FS optimization (only owners download) ────────────────
    async def download_distributed(
        self, *, window: int | None = None, prefer_highest_staked: bool = True
    ) -> Path | None:
        """
        Distributed download for a SHARED filesystem:
          • rank‑0 downloads .metadata + JSON sidecars
          • data/shard files assigned by: owner = blake2b(filename) % world_size
          • barrier, then rank‑0 fills any missing files, barrier again
        Result: repo_root/checkpoints/<version>/<window>/... is complete for all ranks.
        """
        if window is None:
            window = await self._discover_latest(
                prefer_highest_staked=prefer_highest_staked
            )
            if window is None:
                return None

        layout = Layout(self.version, window)
        local_dir = self._local_dir(layout)

        world, r = _world(), _rank()
        # Try highest-staked bucket first, then own
        bucket = await self._choose_read_bucket(
            prefer_highest_staked=prefer_highest_staked
        )
        s3 = await self.comms._get_s3_client(bucket)
        # 1) Enumerate keys under the window prefix
        keys: list[str] = []
        cont = None
        while True:
            args = {"Bucket": bucket.name, "Prefix": f"{layout.prefix}/"}
            if cont:
                args["ContinuationToken"] = cont
            resp = await s3.list_objects_v2(**args)
            for o in resp.get("Contents", []) or []:
                if k := o.get("Key"):
                    keys.append(k)
            if not resp.get("IsTruncated"):
                break
            cont = resp.get("NextContinuationToken")

        # 2) Partition work
        assigned: list[str] = []
        for key in keys:
            fname = Path(key).name
            if (r == 0 and _is_meta(Path(fname))) or (
                not _is_meta(Path(fname)) and _owner_for(fname, world) == r
            ):
                assigned.append(key)

        # 3) Owners download their files (skip if present)
        for key in assigned:
            dst = self.repo_root / key  # mirrored path
            if dst.exists():
                continue
            await self.comms.s3_get_object(
                key=key, bucket=bucket, load_data=False, show_progress=False
            )

        _barrier()  # all owners finished

        # 4) Rank‑0 mop‑up (if any file is still missing, download it)
        if r == 0:
            for key in keys:
                dst = self.repo_root / key
                if not dst.exists():
                    await self.comms.s3_get_object(
                        key=key, bucket=bucket, load_data=False, show_progress=False
                    )

        _barrier()  # ensure folder is complete for all ranks
        return local_dir

    # ── Load (reshard automatically to current topology) ───────────────────────
    def load_local(
        self, *, model, window: int, process_group: dist.ProcessGroup | None = None
    ) -> None:
        layout = Layout(self.version, window)
        ckpt_dir = self._local_dir(layout)
        state = {"app": AppState(model)}
        load(state_dict=state, checkpoint_id=str(ckpt_dir), process_group=process_group)

    async def download_and_load(
        self,
        *,
        model,
        window: int | None = None,
        shared_fs: bool = True,
        process_group: dist.ProcessGroup | None = None,
        prefer_highest_staked: bool = True,
    ) -> int | None:
        local_dir = await (
            self.download_distributed(
                window=window, prefer_highest_staked=prefer_highest_staked
            )
            if shared_fs
            else self.download_all(
                window=window, prefer_highest_staked=prefer_highest_staked
            )
        )
        if local_dir is None:
            return None
        sidecar = json.loads((local_dir / "extra_metadata.json").read_text())
        w = int(sidecar["window"])
        self.load_local(model=model, window=w, process_group=process_group)
        return w
