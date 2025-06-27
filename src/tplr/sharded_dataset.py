import glob, gc
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.distributed as dist


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
            dist.barrier()

        share_list = [None]  # container for broadcast

        # Step1: load data to rank 0
        if self.rank == 0:
            files = sorted(Path(shards_path).glob("train_*.npy"))
            if not files:
                raise FileNotFoundError(f"No train_*.npy shards in {shards_path}")

            tokens_np = np.concatenate([np.load(f).astype(np.int32) for f in files])
            tokens = torch.from_numpy(tokens_np).share_memory_()  # one copy in RAM
            del tokens_np, files
            gc.collect()
            share_list[0] = tokens

        ## Step2: broadcast the shared tensor to all ranks
        if self.world > 1:
            dist.broadcast_object_list(share_list, src=0)
            dist.barrier()

        self.tokens = share_list[0]  # shared tensor for all ranks
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
