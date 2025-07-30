#!/usr/bin/env python
"""
Local multi-GPU smoke test for miner.py.

Runs several inner-loop optimisation windows on real data (bins under
DATASET_BINS_PATH) instead of random tensors. Works out-of-the-box with
`torchrun --nproc_per_node=2 ...`. Prints only on the global root rank.
Boots the Llama-3 8B model in TorchTitan and lets you override the most
important hparams from the command line.
"""
from __future__ import annotations

import builtins, os, sys, types, argparse, asyncio, json, random, math
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Root-rank printing helper
_ORIG_PRINT = builtins.print

def _is_root_rank() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0

def _print(*a, **kw):
    if _is_root_rank():
        _ORIG_PRINT(*a, **kw)

builtins.print = _print

# Command-line arguments
argp = argparse.ArgumentParser()
argp.add_argument("--device",        default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
argp.add_argument("--amp-dtype",     default="bf16", choices=["bf16", "fp16"])
argp.add_argument("--sequence-length", type=int,     default=2048)
argp.add_argument("--micro-batch-size", type=int,    default=2)
argp.add_argument("--inner-windows", type=int,       default=500, help="Number of Miner.inner_steps() windows to run")
argp.add_argument("--num-workers",   type=int,       default=2, help="DataLoader workers")
args = argp.parse_args()

# Environment bootstrapping & dummy creds
print("Setting dummy environment variables for local testing")
_DUMMY_VARS = {
    "R2_GRADIENTS_ACCOUNT_ID":          "dummy_id",
    "R2_GRADIENTS_BUCKET_NAME":         "dummy_bucket",
    "R2_GRADIENTS_READ_ACCESS_KEY_ID":  "dummy_key",
    "R2_GRADIENTS_READ_SECRET_ACCESS_KEY": "dummy_secret",
    "R2_GRADIENTS_WRITE_ACCESS_KEY_ID": "dummy_key",
    "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY": "dummy_secret",
    "R2_AGGREGATOR_ACCOUNT_ID":         "dummy_id",
    "R2_AGGREGATOR_BUCKET_NAME":        "dummy_bucket",
    "R2_AGGREGATOR_READ_ACCESS_KEY_ID":    "dummy_key",
    "R2_AGGREGATOR_READ_SECRET_ACCESS_KEY":"dummy_secret",
    "R2_DATASET_ACCOUNT_ID":            "dummy_id",
    "R2_DATASET_BUCKET_NAME":           "dummy_bucket",
    "R2_DATASET_READ_ACCESS_KEY_ID":    "dummy_key",
    "R2_DATASET_READ_SECRET_ACCESS_KEY":"dummy_secret",
    "DATASET_BINS_PATH":                "/workspace/templar/scripts/dataset_prep/test_shard",
}
for k, v in _DUMMY_VARS.items():
    os.environ.setdefault(k, v)

# Ensure single-node defaults if the launcher forgot them
os.environ.setdefault("RANK",        "0")
os.environ.setdefault("WORLD_SIZE",  "1")
os.environ.setdefault("LOCAL_RANK",  os.environ["RANK"])

# Fake/stub out external services before importing Miner
try:
    import bittensor as bt
except ModuleNotFoundError:
    bt = types.ModuleType("bittensor")
    sys.modules["bittensor"] = bt

class _FakeHotkey:
    ss58_address = "fake_hotkey"

class _FakeWallet:
    def __init__(self, *_, **__):
        self.hotkey = _FakeHotkey()
    @staticmethod
    def add_args(_): pass

bt.wallet = _FakeWallet

class _FakeMetagraph:
    def __init__(self, hotkey):
        self.hotkeys = [hotkey]
        self.S       = [1.0]
        self.netuid  = 268

class _FakeSubtensor:
    block = 123_456
    def __init__(self, *_, **__): pass
    @staticmethod
    def add_args(_): pass
    def metagraph(self, _): return _FakeMetagraph(_FakeHotkey().ss58_address)

bt.subtensor = _FakeSubtensor

import importlib
tplr = importlib.import_module("tplr")

class _FakeComms:
    def __init__(self, *_, **__): self.peers = []
    def get_own_bucket(self, *_, **__): return None
    def try_commit(self, *_, **__):     pass
    def start_commitment_fetcher(self): pass

tplr.comms.Comms      = _FakeComms
tplr.initialize_wandb = lambda *_a, **_k: types.SimpleNamespace(log=lambda *_, **__: None)
tplr.metrics          = types.SimpleNamespace(MetricsLogger=lambda *_a, **_k: None)

# TorchTitan Llama-3 8B hparams
_TORCHTITAN_HPARAMS = {
    "spec_version": 5,
    "project": "dough",
    "sequence_length": 2048,
    "micro_batch_size": 1,
    "target_batch_size": 1,
    "batch_size": 128,
    "inner_steps": 30,
    "inner_learning_rate": 2e-4,
    "outer_learning_rate": 0.9,
    "blocks_per_window": 4096,
    "windows_per_weights": 7,
    "momentum_decay": 0.999,
    "topk_compression": 32,
    "target_chunk": 64,
    "use_dct": False,
    "binary_score_ma_alpha": 0.05,
    "moving_average_window": 5,
    "model_size": "8B",
    "weight_decay": 0.1,
    "warmup_steps": 750,
    "alpha_f": 0.1,
    "t_max": 20000,
    "validator_offset": 1,
    "checkpoint_frequency": 5,
    "max_topk_peers": 15,
    "minimum_peers": 5,
    "peer_replacement_frequency": 5,
    "peer_list_window_margin": 1,
    "active_check_interval": 300,
    "recent_windows": 5,
    "power_normalisation": 2.0,
    "validator_sample_micro_bs": 4,
    "gather_peers_slash_threshold": 0.4,
    "uids_per_window": 20,
    "time_window_delta_seconds": 30,
    "reset_inactivity_windows": 10,
    "sync_max_steps_behind": 3,
    "eval_lr_factor": 0.5,
    "openskill_beta": 7,
    "openskill_tau": 0.1,
    "num_evaluation_bins": 5,
    "quantization_bins": 256,
    "quantization_range": 6,
    "burn_rate": 0.8,
    "idx_overlap_threshold": 0.5,
    "torchtitan": {
        "tp_degree": 2, "dp_replicate": 1, "pp_degree": 1, "cp_degree": 1,
        "compile": False,
        "enable_cpu_offload": False,
        "mixed_precision_param": "float32",
        "mixed_precision_reduce": "float32",
        "enable_async_tensor_parallel": False,
        "disable_loss_parallel": True,
        "fsdp_reshard_after_forward": "default",
        "enable_compiled_autograd": False,
        "float8_recipe_name": None,
        "activation_checkpoint": {"mode": "selective", "option": "op"},
    },
}
_hparams_file = NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump(_TORCHTITAN_HPARAMS, _hparams_file)
_hparams_file.flush()

# Prepare CLI argv for Miner and import it
sys.argv = [
    "local_miner_smoke",
    "--local",
    "--device", args.device,
    "--amp-dtype", args.amp_dtype,
    "--hparams-file", _hparams_file.name,
]

print("Initialising Miner (Llama-3 8B / TorchTitan)…")
from neurons.miner import Miner
miner = Miner()
miner.model.eval()
print("Miner initialised.")

# Dataset loader for .bin token shards
class BinShardDataset(Dataset):
    """
    Expects a directory of uint16-encoded `.bin` files containing
    contiguous token IDs. Each item is a (seq_len,) int64 tensor.
    """
    def __init__(self, root: str | Path, seq_len: int):
        self.seq_len = seq_len
        self.files   = sorted(Path(root).glob("*.bin"))
        if not self.files:
            raise FileNotFoundError(f"No .bin files found under {root}")
        self.maps  = [np.memmap(f, dtype=np.uint16, mode="r") for f in self.files]
        self.counts = [len(m) // seq_len for m in self.maps]
        self.cum_counts = np.cumsum([0] + self.counts)

    def __len__(self): return self.cum_counts[-1]

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx = np.searchsorted(self.cum_counts, idx, side="right") - 1
        local_idx = idx - self.cum_counts[file_idx]
        start = local_idx * self.seq_len
        arr = self.maps[file_idx][start : start + self.seq_len]
        return torch.tensor(arr, dtype=torch.long)

_dataset_root = Path(os.environ["DATASET_BINS_PATH"])
dataset = BinShardDataset(_dataset_root, args.sequence_length)
loader  = DataLoader(
    dataset,
    batch_size=args.micro_batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=(args.device.startswith("cuda")),
    drop_last=True,
)

print(f"Dataset loaded from '{_dataset_root}' – {len(dataset):,} sequences")

# Run Miner.inner_steps() for several windows
async def _run():
    for window in range(args.inner_windows):
        # Run the inner loop (gradients for outer step are produced)
        stats = await miner.inner_steps(loader=loader, step_window=window)

        # Apply the outer optimiser (updates model weights)
        miner.outer_optimizer.step()
        miner.outer_optimizer.zero_grad(set_to_none=True)

        # Report metrics
        print(f"\n[Window {window}] metrics:")
        print(json.dumps(stats, indent=2))

print(f"Running {args.inner_windows} inner-windows…")
asyncio.run(_run())

print("Finished.")