#!/usr/bin/env python
"""
Local smoke test for miner.py
Runs one inner-loop optimisation pass on dummy data, without touching
Subtensor / R2 / WandB. Future miner-side changes are picked up automatically.
"""
from __future__ import annotations
import os, sys, types, argparse, asyncio, json, random
from pathlib import Path

# Set dummy environment variables
# This is necessary because `tplr.config` reads these variables upon import.
print("Setting dummy environment variables for local testing")
dummy_vars = {
    "R2_GRADIENTS_ACCOUNT_ID": "dummy_id",
    "R2_GRADIENTS_BUCKET_NAME": "dummy_bucket",
    "R2_GRADIENTS_READ_ACCESS_KEY_ID": "dummy_key",
    "R2_GRADIENTS_READ_SECRET_ACCESS_KEY": "dummy_secret",
    "R2_GRADIENTS_WRITE_ACCESS_KEY_ID": "dummy_key",
    "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY": "dummy_secret",
    "R2_AGGREGATOR_ACCOUNT_ID": "dummy_id",
    "R2_AGGREGATOR_BUCKET_NAME": "dummy_bucket",
    "R2_AGGREGATOR_READ_ACCESS_KEY_ID": "dummy_key",
    "R2_AGGREGATOR_READ_SECRET_ACCESS_KEY": "dummy_secret",
    "R2_DATASET_ACCOUNT_ID": "dummy_id",
    "R2_DATASET_BUCKET_NAME": "dummy_bucket",
    "R2_DATASET_READ_ACCESS_KEY_ID": "dummy_key",
    "R2_DATASET_READ_SECRET_ACCESS_KEY": "dummy_secret",
    "DATASET_BINS_PATH": "/workspace/templar/scripts/dataset_prep/test_shard",
}
for key, value in dummy_vars.items():
    os.environ.setdefault(key, value)

# Fake/stub external services before importing miner.py
# If bittensor isn't installed, create a dummy module; otherwise monkey-patch.
try:
    import bittensor as bt
except ModuleNotFoundError:
    bt = types.ModuleType("bittensor")
    sys.modules["bittensor"] = bt

# Stub wallet
class _FakeHotkey:
    def __init__(self): 
        self.ss58_address = "fake_hotkey"

class _FakeWallet:
    @staticmethod
    def add_args(parser): 
        pass
    
    def __init__(self, *a, **kw): 
        self.hotkey = _FakeHotkey()

bt.wallet = _FakeWallet

# Stub subtensor & metagraph
class _FakeMetagraph:
    def __init__(self, hotkey): 
        self.hotkeys = [hotkey]
        self.S = [1.0]
        self.netuid = 268

class _FakeSubtensor:
    block = 123_456
    
    @staticmethod
    def add_args(parser): 
        pass
    
    def __init__(self, *a, **kw): 
        pass
    
    def metagraph(self, _netuid): 
        return _FakeMetagraph(_FakeHotkey().ss58_address)

bt.subtensor = _FakeSubtensor

# Patch tplr telemetry (Comms / WandB / Influx) to inert stubs
import importlib
tplr = importlib.import_module("tplr")

class _FakeComms:
    def __init__(self, *a, **kw): 
        self.peers = []
    
    def get_own_bucket(self, *a, **kw): 
        return None
    
    def try_commit(self, *a, **kw): 
        pass
    
    def start_commitment_fetcher(self): 
        pass

tplr.comms.Comms = _FakeComms
tplr.initialize_wandb = lambda *a, **k: types.SimpleNamespace(log=lambda *a, **k: None)
tplr.metrics = types.SimpleNamespace(MetricsLogger=lambda *a, **k: None)

# CLI for this script
argp = argparse.ArgumentParser()
argp.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu")
argp.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
args = argp.parse_args()

# Ensure single-process env
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

# Additional dummy env vars so tplr.config doesn't complain about R2 creds
for k in (
    "R2_GRADIENTS_ACCOUNT_ID", "R2_GRADIENTS_BUCKET_NAME",
    "R2_GRADIENTS_READ_ACCESS_KEY_ID", "R2_GRADIENTS_READ_SECRET_ACCESS_KEY",
    "R2_GRADIENTS_WRITE_ACCESS_KEY_ID", "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY",
):
    os.environ.setdefault(k, "dummy")

# Import Miner after stubs are ready
from neurons.miner import Miner
import torch
from torch.utils.data import DataLoader

# Build fake sys.argv for Miner.miner_config() (needs --local flag)
sys.argv = [
    "local_miner_smoke",
    "--local",
    "--device", args.device,
    "--amp-dtype", args.amp_dtype,
]

# Instantiate Miner (toy model, small hparams)
print("Initialising Miner (toy model)...")
miner = Miner()
miner.model.eval()
print("Miner initialised.")

# Create dummy input data that matches the tokenizer vocab
seq_len = miner.hparams.sequence_length
bs = miner.hparams.micro_batch_size
vocab = miner.tokenizer.vocab_size
torch.manual_seed(0)
dummy = torch.randint(0, vocab, (bs * 10, seq_len), dtype=torch.long)
loader = DataLoader(dummy, batch_size=bs)

# Run one inner-steps optimisation pass
async def _run():
    result = await miner.inner_steps(loader=loader, step_window=0)
    print("\nMetrics returned by inner_steps:")
    print(json.dumps(result, indent=2))

print("Running inner_steps on dummy data...")
asyncio.run(_run())