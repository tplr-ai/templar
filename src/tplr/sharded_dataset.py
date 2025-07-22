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
import time
from pathlib import Path
import asyncio

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import tplr


class SharedShardedDataset(Dataset):
    """
    Memory-maps the *pre-processed* dataset produced by `run_preprocessing()`.

    • Zero runtime concatenation or hashing – everything is done offline.
    • All ranks slice the same mmap; no extra copies.
    • Each worker sees ⌈N / world_size⌉ samples (last worker may get fewer).
    """

    def __init__(
        self,
        shard_index,
        sequence_length: int,
        rank: int,
        world_size: int,
        *,
        token_dtype: npt.DTypeLike = np.uint16,  # MUST match preprocessing
    ):
        super().__init__()
        t0 = time.perf_counter()
        self.seqlen = sequence_length

        # Detect DDP context (works in single-process mode too)
        self.rank = rank
        self.world = world_size
        if self.world > 1:
            dist.barrier(device_ids=[self.rank])
            
        self.tokens_file, self.ids_file = self.locate_shards(shard_index)
        _ = self.mmap_tokens_and_ids(self.tokens_file, self.ids_file)      
        
        if not self.tokens_file.exists() or not self.ids_file.exists():
            raise FileNotFoundError(
                f"Pre-processed files not found in {'/'.join(tokens_file.split('/')[:-1])}. "
                "Run the preprocessing script first."
            )

        # should wrap in a timer
        tplr.logger.info(
            f"[Dataset] rank {self.rank}: init done in {time.perf_counter() - t0:.1f}s "
            f"({self.total_samples} samples)"
        )
    
    @staticmethod
    def locate_shards(
        self, 
        shard_index: int,
        custom_path: os.PathLike | None = None, # extend the use of this static method
    ) -> list[Path]:
        # where should we have a miner download to? # NEEDS_DOCS
        default_path = os.getenv("DATASET_BINS_PATH")
        if default_path is None:
            # this is redundant because of config anyway
            raise ValueError("dataset path not configured. Set $DATASET_BINS_PATH") # DATASET_BINS_PATH is local path?
        
        shards_path = custom_path or default_path
        
        tokens_file = os.path.join(shards_path, f'shard_{shard_index:06d}.npy')
        ids_file = tokens_file.replace('.npy', '.ids')
        
        return tokens_file, ids_file
    
    def mmap_tokens_and_ids(self, token_dtype, tokens_file, ids_file):
        # ────────────────────────── mmap tokens & ids ───────────────────────────
        # Normalise once for safety; still type-checks
        tokens_mem = np.memmap(tokens_file, dtype=np.dtype(token_dtype), mode="r+")
        tokens_mem.flags.writeable = True
        self.tokens = torch.from_numpy(tokens_mem)
        tokens_mem.flags.writeable = False

        ids_mem = np.memmap(ids_file, dtype=np.uint64, mode="r+")
        ids_mem.flags.writeable = True
        self.sample_ids = torch.from_numpy(ids_mem).to(torch.uint64)
        ids_mem.flags.writeable = False

        self.total_samples = len(self.sample_ids)
        return    

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx >= self.total_samples:
            raise IndexError
        start = idx * self.seqlen
        end = start + self.seqlen
        return self.tokens[start:end]

    def sample_id(self, idx: int) -> int:
        return int(self.sample_ids[idx].item()
        

# should we just inherit? since I use the cheeky locate_shards?
class ShardedDatasetManager:
    def __init__(
        self,
        local_dataset_path: str,
        sequence_length: int,
        rank: int,
        world_size: int,
        comms: tplr.comms.Comms,
        token_dtype: npt.DTypeLike = np.uint16,
    ):
        self.local_dataset_path = Path(local_dataset_path)
        self.sequence_length = sequence_length
        self.rank = rank
        self.world_size = world_size
        self.token_dtype = token_dtype
        
        # can we do awaits in regular fn def / inits?
        self.active_dataset: SharedShardedDataset | None = None
        self.upcoming_dataset: SharedShardedDataset | None = None
        
        self.comms = comms
        
        # should comms glob to know all file paths?
        # self.max_dataset_idx = bucket_glob_files_idx
    
    async def prepare_shard(self, shard_index: int):
        download_completed = True
        tokens_file, ids_file = SharedShardedDataset.locate_shards(shard_index)
        tplr.logger.info(f"Preparing shard {shard_index} at {shard_path}")
        
        if not os.path.exists(shard_path):
            bucket = self.comms.get_own_bucket("shared_dataset", "read")
            # don't have this be await 
            download_completed = asyncio.create_task(
                self.comms.s3_get_object(
                    filename,
                    bucket,
                ),
            )
        return download_completed
    
    async def create_dataloader(self, shard_index):
        # this await may be sometimes redundant?
        downloaded = await prepare_shard(shard_index)
        dataset = SharedShardedDataset(
            shard_index=shard_index,
            sequence_length=self.sequence_length,
            rank=self.rank,
            world_size=self.world_size,
            token_dtype=self.token_dtype,
        )
        return dataset
    
    async def initialize_datasets(self, current_shard_index: int):
        # await this one so dataset does exist, the check if the files not found
        # would raise an error
        self.active_dataset = await self.set_current_dataloader(current_shard_index)
        # it's possible to create_task like this?
        self.upcoming_dataset = asyncio.create_task(self.prepare_shard(current_shard_index + 1))
    
    async def swap_datasets(self):
        # How would we create async task to track this?
        if self.upcoming_dataset and not self.upcoming_dataset.done():
            await self.upcoming_dataset
        
        if self.upcoming_dataset is None:
            # end of training shards, restart?
            # more like pass incremented shards and see
            # if > max_dataset_idx
            pass
        
        self.shard_index += 1
        old_dataset = self.active_dataset
        self.active_dataset = create_dataloader(self.shard_index)
        self.upcoming_dataset = None
        
        tplr.logger.info("successfully swapped datasets.")
        
        if old_dataset:
            # os.remove(old_dataset.tokens_file)
            # os.remove(old_dataset.ids_file)
            # YES DESTROY, need to think that through more
            old_dataset.destroy()
            