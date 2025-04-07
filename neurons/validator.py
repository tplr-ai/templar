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

# Standard library
import argparse
import asyncio
import copy
import os
import random
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from io import StringIO
from time import perf_counter

import bittensor as bt
import numpy as np

# Third party
import torch
from rich.console import Console
from rich.table import Table
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from transformers import LlamaForCausalLM

# Local
import tplr

# GPU optimizations.
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Import the MoE model.
from tplr.moe_model import MoE

@contextmanager
def timer(name: str, wandb_obj=None, step=None, metrics_logger=None):
    start = perf_counter()
    yield
    duration = perf_counter() - start
    tplr.logger.debug(f"{name} took {duration:.2f}s")
    if wandb_obj and step is not None:
        wandb_obj.log({f"validator/{name}": duration}, step=step)
    if metrics_logger and step is not None:
        metrics_logger.log(
            measurement="timing", tags={"window": step}, fields={name: duration}
        )


class Validator:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description="MoE Validator script")
        parser.add_argument("--netuid", type=int, default=268, help="Bittensor network UID.")
        parser.add_argument("--project", type=str, default="templar", help="Wandb project.")
        parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
        parser.add_argument("--num_experts", type=int, default=4, help="Number of experts in the MoE model")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        config = bt.config(parser)
        return config

    def __init__(self):
        self.config = Validator.config()
        self.hparams = {
            "d_model": 128,
            "nhead": 8,
            "num_layers": 2,
            "vocab_size": 10000,
            "learning_rate": 0.001,
            "num_experts": self.config.num_experts,
            "momentum_decay": 0.9,
            "topk_compression": 10,
            "ema_decay": 0.9
        }
        self.device = self.config.device
        # Initialize bittensor objects.
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(int(self.config.netuid))
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            tplr.logger.error("Wallet not registered.")
            sys.exit()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Instantiate the MoE model.
        base_config = {
            "d_model": self.hparams["d_model"],
            "nhead": self.hparams["nhead"],
            "num_layers": self.hparams["num_layers"],
            "vocab_size": self.hparams["vocab_size"],
        }
        self.model = MoE(base_config, num_experts=self.hparams["num_experts"])
        self.model.to(self.device)

        # Initialize optimizer and momentum dictionary for expert modules.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams["learning_rate"])
        self.momentum = {}
        for expert_id in range(self.hparams["num_experts"]):
            for name, param in self.model.experts[expert_id].named_parameters():
                key = f"expert_{expert_id}.{name}"
                self.momentum[key] = torch.zeros_like(param)

        # Initialize compression and comms objects.
        self.transformer = tplr.compress.TransformDCT(self.model, target_chunk=16)
        self.compressor = tplr.compress.CompressDCT()
        self.comms = tplr.comms.Comms(
            self.wallet, config=self.config, metagraph=self.metagraph, hparams=self.hparams, uid=self.uid
        )

        # Prepare EMA scores for each expert.
        self.expert_scores = {i: 0.0 for i in range(self.hparams["num_experts"])}

    async def evaluate_expert(self, expert_id, validation_data):
        # For demonstration, generate dummy validation data.
        seq_len = 10
        batch_size = 16
        dummy_input = torch.randn(seq_len, batch_size, self.hparams["d_model"]).to(self.device)
        dummy_targets = torch.randint(0, self.hparams["vocab_size"], (batch_size,), device=self.device)
        logits, gating_weights, expert_indices = self.model(dummy_input)
        loss = torch.nn.functional.cross_entropy(logits, dummy_targets)
        return loss.item()

    async def run(self):
        num_steps = 50  # Demonstration steps.
        for step in range(num_steps):
            # Simulate receiving expert gradients from peers.
            received_gradients = {}
            for expert_id in range(self.hparams["num_experts"]):
                grad_packet = await self.comms.get(key=f"expert_gradient_{expert_id}")
                if grad_packet is not None:
                    received_gradients[expert_id] = grad_packet
            
            # For each expert, evaluate the quality of the received gradient.
            for expert_id, grad_packet in received_gradients.items():
                expert_module = self.model.experts[expert_id]
                # Save current state.
                backup_state = copy.deepcopy(expert_module.state_dict())
                original_loss = await self.evaluate_expert(expert_id, validation_data=None)
                
                # Decompress the gradient.
                decoded_gradient = self.transformer.decode(
                    self.compressor.batch_decompress(
                        next(iter(expert_module.parameters())),
                        grad_packet["idxs"],
                        grad_packet["vals"],
                        grad_packet["xshape"],
                        grad_packet["totalk"]
                    )
                )
                # Apply a simulated sign-based update.
                for name, param in expert_module.named_parameters():
                    param.data.add_(-self.hparams["learning_rate"], decoded_gradient)
                
                updated_loss = await self.evaluate_expert(expert_id, validation_data=None)
                improvement = original_loss - updated_loss
                self.expert_scores[expert_id] = (
                    self.hparams["ema_decay"] * self.expert_scores[expert_id] +
                    (1 - self.hparams["ema_decay"]) * improvement
                )
                # Roll back the expert's state.
                expert_module.load_state_dict(backup_state)
                print(f"Expert {expert_id} step {step}: orig_loss = {original_loss:.4f}, updated_loss = {updated_loss:.4f}, improvement = {improvement:.4f}")
            await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(Validator().run())
