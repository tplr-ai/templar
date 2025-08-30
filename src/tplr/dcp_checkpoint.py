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
import json
import re
import time
from concurrent.futures import Future
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
from torch.distributed.checkpoint.state_dict_saver import async_save, save
from torch.distributed.checkpoint.stateful import Stateful

import tplr


# ── Model-only Stateful (Titan-compatible distributed state dicts) ─────────────
class AppState(Stateful):
    def __init__(self, model):
        self.model = model

    def state_dict(self) -> dict[str, ValueType]:
        return get_model_state_dict(self.model)

    def load_state_dict(self, state: dict[str, ValueType]) -> None:
        set_model_state_dict(self.model, state)


class SnapshotState(Stateful):
    """Immutable snapshot of a model's distributed state dict (CPU-offloaded)."""

    def __init__(self, model):
        # Offload shards to CPU to decouple from in-flight training updates.
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        self._snap = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=False, cpu_offload=True),
        )

    def state_dict(self) -> dict[str, ValueType]:
        return self._snap

    # load_state_dict is never called for saving; keep a no-op to satisfy interface.
    def load_state_dict(self, state: dict[str, ValueType]) -> None:
        return


# ── Utils ─────────────────────────────────────────────────────────────────────
def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _is_meta(p: str | Path) -> bool:
    """True if a filename/key refers to metadata (DCP .metadata or *.json sidecars)."""
    n = Path(p).name.lower()
    return n == ".metadata" or n.endswith(".metadata") or n.endswith(".json")


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def _mb(b: int) -> float:
    return b / (1024 * 1024) if b else 0.0


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

        # Optional CPU process group for async_save; created lazily.
        self._cpu_pg: dist.ProcessGroup | None = None
        if dist.is_available() and dist.is_initialized():
            try:
                self._cpu_pg = dist.new_group(
                    ranks=list(range(_world())), backend=dist.Backend.GLOO
                )
                tplr.logger.info(
                    f"[DCP] created CPU(Gloo) process group for async_save on rank {_rank()}"
                )
            except Exception as e:
                tplr.logger.warning(f"[DCP] failed to create CPU(Gloo) PG: {e}")

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

        t0 = time.perf_counter()
        tplr.logger.info(
            f"[DCP][save] rank {_rank()}/{_world()} → begin local save "
            f"(window={window}, dir={out_dir})"
        )
        save(
            state_dict=state,
            checkpoint_id=str(out_dir),
            process_group=process_group,
        )
        dt = time.perf_counter() - t0

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

        # Post‑barrier: count files/bytes that exist on disk (best‑effort)
        try:
            files = [p for p in out_dir.iterdir() if p.is_file()]
            total_bytes = sum(p.stat().st_size for p in files)
            tplr.logger.info(
                f"[DCP][save] rank {_rank()}/{_world()} ← done in {dt:.2f}s "
                f"(~{len(files)} files, ~{_mb(total_bytes):.2f} MiB)"
            )
        except Exception:
            tplr.logger.info(
                f"[DCP][save] rank {_rank()}/{_world()} ← done in {dt:.2f}s"
            )

        return out_dir

    # ── Save (async with snapshot & CPU PG) ───────────────────────────────────
    @dataclass(slots=True)
    class SaveHandle:
        future: Future
        out_dir: Path
        window: int
        t0: float

        async def wait(self) -> None:
            # Avoid blocking the event loop by running the blocking
            # Future.result() call in a worker thread.
            await asyncio.to_thread(self.future.result)

    async def save_local_async(
        self,
        *,
        model,
        window: int,
        sync_window: int,
        topology: str = "FSDP",
    ) -> "DCPCheckpointer.SaveHandle":
        """
        Take a CPU‑offloaded snapshot and kick off an async DCP save.
        Returns a handle you can await before uploading.
        """
        layout = Layout(self.version, window)
        out_dir = self._local_dir(layout)

        # Prepare a stable snapshot (so training can keep going).
        tplr.logger.info(
            f"[DCP][save-async] rank {_rank()}/{_world()} snapshot → (window={window}, dir={out_dir})"
        )
        snap = SnapshotState(model)

        # Write the small sidecar immediately.
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

        # Launch async save using a CPU-enabled PG when available.
        t0 = time.perf_counter()
        pg = self._cpu_pg if self._cpu_pg is not None else None
        fut = async_save(
            state_dict={"app": snap},
            checkpoint_id=str(out_dir),
            process_group=pg,
        )
        tplr.logger.info(
            f"[DCP][save-async] rank {_rank()}/{_world()} launched (window={window})"
        )
        return DCPCheckpointer.SaveHandle(
            future=fut, out_dir=out_dir, window=window, t0=t0
        )

    # ── Upload (per‑rank; rank‑0 handles metadata and pointer) ─────────────────
    async def upload(
        self,
        *,
        window: int,
        background: bool = False,
        delete_local_on_success: bool = True,
        wait_for: "DCPCheckpointer.SaveHandle | None" = None,
    ) -> None:
        layout = Layout(self.version, window)
        local_dir = self._local_dir(layout)

        async def _do() -> None:
            # If the save was launched async, wait for it to complete first.
            if wait_for is not None:
                tplr.logger.info(
                    f"[DCP][upload] rank {_rank()}/{_world()} waiting for save (window={window})"
                )
                await wait_for.wait()
                tplr.logger.info(
                    f"[DCP][upload] rank {_rank()}/{_world()} save finished (window={window})"
                )

            t_all = time.perf_counter()
            world, r = _world(), _rank()

            # Take a snapshot of what's on disk right now.
            files = [p for p in local_dir.iterdir() if p.is_file()]
            data_files = [q for q in files if not _is_meta(q)]
            meta_files = [q for q in files if _is_meta(q)]

            # Deterministic split: round‑robin by sorted filename.
            data_files_sorted = sorted(data_files, key=lambda p: p.name)
            owned_files = [p for i, p in enumerate(data_files_sorted) if i % world == r]
            owned_bytes = sum(p.stat().st_size for p in owned_files)
            meta_bytes = sum(p.stat().st_size for p in meta_files) if r == 0 else 0
            latest_bytes = 0

            tplr.logger.info(
                f"[DCP][upload] rank {r}/{world} start (window={window}) | "
                f"owned shards: {len(owned_files)} (~{_mb(owned_bytes):.2f} MiB)"
                + (
                    f", meta: {len(meta_files)} (~{_mb(meta_bytes):.2f} MiB)"
                    if r == 0
                    else ""
                )
            )

            async def _timed_put(key: str, path: Path, size: int) -> None:
                t0 = time.perf_counter()
                await self.comms.s3_put_object(key=key, file_path=str(path))
                dt = time.perf_counter() - t0
                tplr.logger.info(
                    f"[DCP][upload] rank {r}/{world} ↑ {key} "
                    f"{_mb(size):.2f} MiB in {dt:.2f}s ({_mb(size) / dt if dt > 0 else 0:.2f} MiB/s)"
                )

            # Rank‑0 uploads metadata first (pointer comes later).
            tasks: list[asyncio.Task] = []
            if r == 0:
                for p in meta_files:
                    tasks.append(
                        asyncio.create_task(
                            _timed_put(f"{layout.prefix}/{p.name}", p, p.stat().st_size)
                        )
                    )

            # All ranks upload their owned shard files.
            for p in owned_files:
                tasks.append(
                    asyncio.create_task(
                        _timed_put(f"{layout.prefix}/{p.name}", p, p.stat().st_size)
                    )
                )

            if tasks:
                await asyncio.gather(*tasks)

            # Publish the version pointer **without any barrier** (best-effort).
            if r == 0:
                tmp = local_dir / "_LATEST.json"
                tmp.write_text(json.dumps({"latest_window": int(window)}, indent=2))
                latest_bytes = tmp.stat().st_size
                await _timed_put(
                    f"checkpoints/{self.version}/_LATEST.json", tmp, latest_bytes
                )
                tplr.logger.info(
                    f"[DCP][upload] rank 0 published pointer for version {self.version} → window {window}"
                )

            # Local cleanup: each rank removes only the files it uploaded.
            if delete_local_on_success:
                # Delete owned data shards for this rank.
                for p in owned_files:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
                # Rank‑0 also cleans up metadata and local pointer file.
                if r == 0:
                    for p in meta_files:
                        try:
                            p.unlink(missing_ok=True)
                        except Exception:
                            pass
                    try:
                        (local_dir / "_LATEST.json").unlink(missing_ok=True)
                    except Exception:
                        pass
                    # Optionally prune the directory if empty.
                    try:
                        next(local_dir.iterdir())
                    except StopIteration:
                        try:
                            local_dir.rmdir()
                        except Exception:
                            pass

            dt_all = time.perf_counter() - t_all
            total_up_bytes = (
                owned_bytes
                + (meta_bytes if r == 0 else 0)
                + (latest_bytes if r == 0 else 0)
            )
            tplr.logger.info(
                f"[DCP][upload] rank {r}/{world} done in {dt_all:.2f}s | "
                f"bytes≈{total_up_bytes} ({_mb(total_up_bytes):.2f} MiB)"
            )

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
        t_all = time.perf_counter()
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
        tplr.logger.info(
            f"[DCP][download-all] rank {_rank()}/{_world()} start "
            f"(window={window}, bucket={bucket.name}, prefix={layout.prefix}/)"
        )
        got_any = False
        total_bytes = 0
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
                    size = int(obj.get("Size", 0))
                    t0 = time.perf_counter()
                    await self.comms.s3_get_object(
                        key=key, bucket=bucket, load_data=False, show_progress=False
                    )
                    dt = time.perf_counter() - t0
                    total_bytes += size
                    tplr.logger.debug(
                        f"[DCP][download-all] rank {_rank()}/{_world()} ↓ {key} "
                        f"{_mb(size):.2f} MiB in {dt:.2f}s ({_mb(size) / dt if dt > 0 else 0:.2f} MiB/s)"
                    )
            if not resp.get("IsTruncated"):
                break
            cont = resp.get("NextContinuationToken")

        dt_all = time.perf_counter() - t_all
        if got_any:
            tplr.logger.info(
                f"[DCP][download-all] rank {_rank()}/{_world()} done in {dt_all:.2f}s | "
                f"bytes≈{total_bytes} ({_mb(total_bytes):.2f} MiB)"
                f" thru≈{_mb(total_bytes) / dt_all if dt_all > 0 else 0:.2f} MiB/s"
            )

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
        t_all = time.perf_counter()
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
        keys: list[tuple[str, int]] = []
        cont = None
        while True:
            args = {"Bucket": bucket.name, "Prefix": f"{layout.prefix}/"}
            if cont:
                args["ContinuationToken"] = cont
            resp = await s3.list_objects_v2(**args)
            for o in resp.get("Contents", []) or []:
                if k := o.get("Key"):
                    keys.append((k, int(o.get("Size", 0))))
            if not resp.get("IsTruncated"):
                break
            cont = resp.get("NextContinuationToken")

        # 2) Partition work (round‑robin by filename for data shards; rank‑0 handles metadata)
        meta_keys: list[tuple[str, int]] = [
            (k, sz) for (k, sz) in keys if _is_meta(Path(k).name)
        ]
        data_keys: list[tuple[str, int]] = [
            (k, sz) for (k, sz) in keys if not _is_meta(Path(k).name)
        ]
        data_keys_sorted = sorted(data_keys, key=lambda pair: Path(pair[0]).name)
        assigned: list[tuple[str, int]] = [
            pair for i, pair in enumerate(data_keys_sorted) if i % world == r
        ]
        if r == 0:
            assigned.extend(meta_keys)  # rank‑0 also pulls metadata/JSON

        owned_bytes = sum(sz for _, sz in assigned if not _is_meta(Path(_).name))
        meta_bytes = sum(sz for _, sz in meta_keys) if r == 0 else 0
        tplr.logger.info(
            f"[DCP][download-dist] rank {r}/{world} start (window={window}, bucket={bucket.name}) | "
            f"assigned: {len(assigned)} (~{_mb(owned_bytes):.2f} MiB)"
            + (f", meta-by-r0: ~{_mb(meta_bytes):.2f} MiB" if r == 0 else "")
        )

        # 3) Owners download their files (skip if present)
        for key, sz in assigned:
            dst = self.repo_root / key  # mirrored path
            if dst.exists():
                tplr.logger.debug(
                    f"[DCP][download-dist] rank {r}/{world} skip (exists) {key}"
                )
                continue
            t0 = time.perf_counter()
            await self.comms.s3_get_object(
                key=key, bucket=bucket, load_data=False, show_progress=False
            )
            dt = time.perf_counter() - t0
            tplr.logger.debug(
                f"[DCP][download-dist] rank {r}/{world} ↓ {key} "
                f"{_mb(sz):.2f} MiB in {dt:.2f}s ({_mb(sz) / dt if dt > 0 else 0:.2f} MiB/s)"
            )

        _barrier()  # all owners finished

        # 4) Rank‑0 mop‑up (if any file is still missing, download it)
        if r == 0:
            mop_files = 0
            mop_bytes = 0
            for key, sz in keys:
                dst = self.repo_root / key
                if not dst.exists():
                    t0 = time.perf_counter()
                    await self.comms.s3_get_object(
                        key=key, bucket=bucket, load_data=False, show_progress=True
                    )
                    dt = time.perf_counter() - t0
                    mop_files += 1
                    mop_bytes += sz
                    tplr.logger.debug(
                        f"[DCP][download-dist] rank 0 mop-up ↓ {key} "
                        f"{_mb(sz):.2f} MiB in {dt:.2f}s ({_mb(sz) / dt if dt > 0 else 0:.2f} MiB/s)"
                    )
            tplr.logger.info(
                f"[DCP][download-dist] rank 0 mop-up completed | files={mop_files}, "
                f"bytes≈{mop_bytes} ({_mb(mop_bytes):.2f} MiB)"
            )

        _barrier()  # ensure folder is complete for all ranks

        dt_all = time.perf_counter() - t_all
        total_assigned = sum(sz for _, sz in assigned) + (meta_bytes if r == 0 else 0)
        tplr.logger.info(
            f"[DCP][download-dist] rank {r}/{world} done in {dt_all:.2f}s | "
            f"bytes≈{total_assigned} ({_mb(total_assigned):.2f} MiB) "
            f"thru≈{_mb(total_assigned) / dt_all if dt_all > 0 else 0:.2f} MiB/s"
        )
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
