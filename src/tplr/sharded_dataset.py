import gc
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import tplr


class SharedShardedDataset(Dataset):
    """
    • Rank 0 concatenates every train_*.npy shard, calls .share_memory_(),
      then broadcasts the tensor handle.
    • All ranks slice that single tensor; no extra copies are made.
    • Each worker gets ⌈N/ world_size⌉ samples (last worker may get fewer).
    • Batches are returned on the given CUDA device.
    """

    def __init__(
        self, shards_path: str, sequence_length: int, rank: int, world_size: int
    ):
        super().__init__()
        self.seqlen = sequence_length

        # Detect DDP context (works in single-process mode too)
        self.rank = rank
        self.world = world_size
        if self.world > 1:
            tplr.logger.info(f"[Dataset] rank {self.rank}: entering initial barrier")
            dist.barrier(device_ids=[self.rank])
            tplr.logger.info(f"[Dataset] rank {self.rank}: exited initial barrier")

        # ────────────────────────── load / create memory-mapped file ──────────────────────────
        shards_dir = Path(shards_path)
        mmap_file = shards_dir / "tokens.bin"

        # ──────────────── rank-0 creates the mmap *atomically* (temp + rename) ───────────────
        if self.rank == 0 and not mmap_file.exists():
            tplr.logger.info(f"[Dataset] rank0: concatenating shards → {mmap_file}")
            files = sorted(shards_dir.glob("train_*.npy"))
            if not files:
                raise FileNotFoundError(f"No train_*.npy shards in {shards_dir}")

            load_start = time.perf_counter()
            tokens_np = np.concatenate([np.load(f).astype(np.int32) for f in files])
            tplr.logger.info(
                f"[Dataset] rank0: loaded {len(files)} shards "
                f"in {time.perf_counter() - load_start:.1f}s"
            )

            tmp_path = mmap_file.with_suffix(".bin.tmp")
            tplr.logger.info(
                f"[Dataset] rank0: writing {tokens_np.nbytes / 1e6:.1f} MB to {tmp_path}"
            )
            tokens_np.tofile(tmp_path)  # 1️⃣ write to temp file

            # 2️⃣ flush & fsync to be safe
            with open(tmp_path, "rb+") as _f:
                _f.flush()
                os.fsync(_f.fileno())

            # 3️⃣ atomic move – if another job won the race, this just replaces tmp file
            os.replace(tmp_path, mmap_file)

            del tokens_np, files
            gc.collect()

        # Ensure the file is written before other ranks touch it
        if self.world > 1:
            dist.barrier(device_ids=[self.rank])

        # Map the file (read-only) on every rank
        num_int32 = mmap_file.stat().st_size // 4
        tokens_mem = np.memmap(mmap_file, dtype=np.int32, mode="r+", shape=(num_int32,))

        tokens_mem.flags.writeable = True  # 1️⃣ make Torch happy
        self.tokens = torch.from_numpy(tokens_mem)
        tokens_mem.flags.writeable = False  # 2️⃣ lock it right back to RO
        self.total_samples = len(self.tokens) // self.seqlen

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx >= self.total_samples:
            raise IndexError
        start = idx * self.seqlen
        end = start + self.seqlen
        batch = self.tokens[start:end]
        return batch
