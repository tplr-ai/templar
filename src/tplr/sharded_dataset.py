# src/tplr/in_memory_gpu_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import math
import os
import glob
import numpy as np
from pathlib import Path
from typing import Literal
import gc
import tplr

class ShardedGPUDataset(Dataset):
    """
    A PyTorch Dataset that efficiently loads tokenized data from .npy shards,
    distributing tokens among DDP workers without excessive memory usage.
    """
    def __init__(self,
                 shards_path: str,
                 token_budget: int,
                 sequence_length: int,
                 rank: int,
                 world_size: int,
                 device: torch.device,
                 shard_token_size: int = 100_000_000, # Expected tokens per .npy shard
                 split: Literal["train"] = "train", # only supports "train" for now
                 reside_in_gpu: bool = False):
        """
        Args:
            shards_path (str): Path to the directory containing .npy token shards.
            token_budget (int): Total number of tokens to be used across all workers.
            sequence_length (int): The length of each sequence to be returned.
            rank (int): The rank of the current DDP process.
            world_size (int): The total number of DDP processes.
            device (torch.device): The CUDA device for this rank (e.g., torch.device("cuda:0")).
            shard_token_size (int): Expected number of tokens in each .npy shard file.
                                    Used to calculate how many shards to load for the budget.
            split (str): Train test split to load data (e.g., "train").
        """
        super().__init__()
        self.shards_path = Path(shards_path)
        self.token_budget = token_budget
        self.sequence_length = sequence_length
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.reside_in_gpu = reside_in_gpu 
        self.shard_token_size = shard_token_size
        self.shard_filename_prefix = f"{split}_"

        if not self.shards_path.is_dir():
            raise FileNotFoundError(f"Shards directory not found: {self.shards_path}")

        # 1. Discover and sort shard files
        shard_files = sorted(glob.glob(str(self.shards_path / f"{self.shard_filename_prefix}*.npy")))
        if not shard_files:
            raise FileNotFoundError(f"No shard files found with prefix '{self.shard_filename_prefix}' in {self.shards_path}")

        # 2. Calculate how many shards to load
        num_shards_to_load = math.ceil(self.token_budget / self.shard_token_size)
        
        if num_shards_to_load > len(shard_files):
            tplr.logger.warning(f"[Rank {self.rank}]: Requested to load {num_shards_to_load} shards, but only {len(shard_files)} are available. Using all available shards.")
            num_shards_to_load = len(shard_files)

        # 3. Load shards incrementally and extract worker's portion
        tplr.logger.info(f"[Rank {self.rank}] Loading {num_shards_to_load} shards incrementally...")
        
        worker_token_chunks = []
        total_tokens_processed = 0
        tokens_per_worker = self.token_budget // self.world_size
        worker_start_token = self.rank * tokens_per_worker
        worker_end_token = (self.rank + 1) * tokens_per_worker
        
        # Ensure last worker gets any remaining tokens
        if self.rank == self.world_size - 1:
            worker_end_token = self.token_budget

        for i in range(num_shards_to_load):
            if total_tokens_processed >= self.token_budget:
                break
                
            shard_file_path = shard_files[i]
            
            try:
                # Load shard data
                shard_data_np = np.load(shard_file_path).astype(np.int32)
                shard_tokens = len(shard_data_np)
                
                # Calculate which part of this shard belongs to current worker
                shard_start_global = total_tokens_processed
                shard_end_global = min(total_tokens_processed + shard_tokens, self.token_budget)
                
                # Check if this shard overlaps with current worker's range
                if shard_end_global > worker_start_token and shard_start_global < worker_end_token:
                    # Calculate local indices within the shard
                    local_start = max(0, worker_start_token - shard_start_global)
                    local_end = min(shard_tokens, worker_end_token - shard_start_global)
                    
                    if local_start < local_end:
                        # Extract worker's portion from this shard
                        worker_portion = shard_data_np[local_start:local_end]
                        worker_token_chunks.append(torch.tensor(worker_portion, dtype=torch.long))
                        
                        tplr.logger.debug(f"[Rank {self.rank}] Shard {i}: extracted tokens {local_start}:{local_end} "
                                        f"(global {shard_start_global + local_start}:{shard_start_global + local_end})")
                
                total_tokens_processed = shard_end_global
                
                # Clean up numpy array immediately
                del shard_data_np
                
            except Exception as e:
                raise IOError(f"Error loading shard file {shard_file_path}: {e}")

        # 4. Concatenate worker's token chunks
        if not worker_token_chunks:
            raise ValueError(f"[Rank {self.rank}] No tokens loaded for this worker. Check token budget and worker assignment.")

        worker_tokens_cpu = torch.cat(worker_token_chunks, dim=0)
        
        # Clean up chunks
        del worker_token_chunks
        gc.collect()
        
        # 5. Move to GPU if requested
        if self.reside_in_gpu:
            self.worker_tokens = worker_tokens_cpu.to(self.device)
            del worker_tokens_cpu  # Free CPU memory
        else:
            self.worker_tokens = worker_tokens_cpu

        # Calculate number of full sequences (samples) for this worker
        self.num_samples = len(self.worker_tokens) // self.sequence_length
        
        actual_tokens = len(self.worker_tokens)
        tplr.logger.info(f"[Rank {self.rank}] Loaded {actual_tokens:,} tokens, "
                        f"creating {self.num_samples:,} samples of length {self.sequence_length}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.sequence_length
        end = start + self.sequence_length
        batch = self.worker_tokens[start:end]
        return batch

def get_sharded_gpu_dataloader(
    dataset: ShardedGPUDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    num_prefetch_batches: int = 2
):
    """
    Creates a PyTorch DataLoader for the ShardedGPUDataset.
    """
    if not isinstance(dataset, ShardedGPUDataset):
        raise TypeError("dataset must be an instance of ShardedGPUDataset")

    if dataset.reside_in_gpu:
        num_workers = 0
        num_prefetch_batches = None
        persistent_workers = False

    pin_memory = num_workers > 0
    persistent_workers = num_workers > 0
    if num_workers == 0 or num_prefetch_batches == 0:
        num_prefetch_batches = None
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=num_prefetch_batches,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory
    )
