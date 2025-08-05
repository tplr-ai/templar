import argparse
import asyncio
import json
import os
from types import SimpleNamespace

# --- Set Dummy Environment Variables ---
# This is necessary because `tplr.config` reads these variables upon import.
# These values are placeholders and will not be used in this local test.
print("--- Setting dummy environment variables for local testing ---")
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
    "DATASET_BINS_PATH": "/tmp/dummy_dataset_path",
}
for key, value in dummy_vars.items():
    os.environ.setdefault(key, value)


import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import LlamaForCausalLM
from contextlib import nullcontext

import tplr
from neurons.miner import Miner

# GPU optimizations
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class LocalTestMiner(Miner):
    """
    A subclass of Miner that overrides network-dependent initializations
    and the training step to work with a standard PyTorch DataLoader.
    """
    def __init__(self, config):
        # We manually set up only what's necessary for the training loop,
        # skipping the original Miner.__init__ and its network setup.
        
        # Basic configuration
        self.config = config
        self.device = torch.device(config.device)
        self.is_master = True
        self.rank = 0
        self.world_size = 1

        # Load hparams for a local "toy" model
        self.hparams = tplr.load_hparams(use_local_run_hparams=True)
        
        # Override batch sizes for a quick, predictable test
        # `batch_size` is the effective batch size after gradient accumulation.
        # `micro_batch_size` is the size processed in a single forward/backward pass.
        self.hparams.batch_size = 4
        self.hparams.micro_batch_size = 2
        
        # Model and Tokenizer setup
        self.model = LlamaForCausalLM(self.hparams.model_config)
        self.model.to(self.device)
        self.tokenizer = self.hparams.tokenizer

        # Simplified Optimizers for local test
        self.inner_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.inner_learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        self.inner_scheduler = torch.optim.lr_scheduler.StepLR(self.inner_optimizer, step_size=1, gamma=0.9)
        
        # Mock necessary attributes used in inner_steps
        self.stop_event = asyncio.Event()
        self.loop = asyncio.get_running_loop()
        self.current_window = 0
        
        # The inner_steps method requires a sampler with `grad_accum_steps`
        grad_accum_steps = self.hparams.batch_size // (self.hparams.micro_batch_size * self.world_size)
        self.sampler = SimpleNamespace(grad_accum_steps=grad_accum_steps)
        print(f"Gradient Accumulation Steps: {self.sampler.grad_accum_steps}")

    def should_continue(self, local_has_batch: bool, device) -> bool:
        """Override network sync with a simple local check."""
        return local_has_batch

    def _ddp_reduce(self, value: torch.Tensor | float, op=None) -> float:
        """Override DDP reduction for single-process run."""
        if isinstance(value, torch.Tensor):
            return value.item()
        return float(value)

    def _get_offloaded_param(self):
         """Mock parameter offloading."""
         return [p.data.detach().clone().to("cpu") for p in self.model.parameters()]

    # --- Override the inner_steps method to handle the dummy data loader ---
    async def inner_steps(
        self,
        loader: DataLoader,
        step_window: int,
    ) -> dict:
        """
        This is an override of the original `inner_steps` method.
        The only change is to correctly unpack the batch from the dummy DataLoader.
        """
        self.inner_optimizer.zero_grad()
        total_loss, batch_count, batch_tokens = 0.0, 0, 0
        
        # Get a "before" snapshot of the model weights for the pseudo-outer-step
        params_offloaded = self._get_offloaded_param()

        for i, batch in enumerate(loader):
            # THE FIX IS HERE: Unpack the tensor from the list yielded by DataLoader
            input_ids = batch[0].to(self.device)
            labels = input_ids.clone()

            # The rest of this loop is the same as the original inner_steps
            tokens_this_batch = input_ids.numel()
            batch_tokens += tokens_this_batch
            
            # Forward + backward pass
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                outputs = self.model(input_ids=input_ids, labels=labels)

            loss = outputs.loss / self.sampler.grad_accum_steps
            loss.backward()
            total_loss += loss.detach().item() * self.sampler.grad_accum_steps
            batch_count += 1
            
            # Optimizer step after accumulating gradients
            if (i + 1) % self.sampler.grad_accum_steps == 0:
                print(f"  -> Optimizer step at batch {i+1}")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.inner_optimizer.step()
                self.inner_scheduler.step()
                self.inner_optimizer.zero_grad()
        
        # Recreate the "pseudo-gradient" for the outer step, as in the original method
        with torch.no_grad():
            for saved_param, p in zip(params_offloaded, self.model.parameters()):
                saved_param = saved_param.to(p.device, non_blocking=True)
                p.grad = saved_param - p.data
                p.data.copy_(saved_param)

        return {"total_loss": total_loss, "batch_count": batch_count, "batch_tokens": batch_tokens, "window_entry_loss": total_loss / batch_count if batch_count > 0 else 0}


async def main():
    """Main function to initialize and run the test."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    config = parser.parse_args()

    print("--- Initializing LocalTestMiner ---")
    miner = LocalTestMiner(config)
    print(f"Device: {miner.device}")
    print(f"Model parameters: {sum(p.numel() for p in miner.model.parameters()):,}")
    print("--- Miner Initialized ---\n")

    # --- Create Dummy Data ---
    sequence_length = miner.hparams.sequence_length
    micro_batch_size = miner.hparams.micro_batch_size
    vocab_size = miner.hparams.tokenizer.vocab_size
    
    num_dummy_samples = micro_batch_size * 10
    dummy_input_ids = torch.randint(0, vocab_size, (num_dummy_samples, sequence_length), dtype=torch.long)
    dummy_dataset = TensorDataset(dummy_input_ids)
    dummy_loader = DataLoader(dummy_dataset, batch_size=micro_batch_size)

    print("--- Inspecting Model I/O Structure ---")
    first_batch = next(iter(dummy_loader))
    print_io_structure(miner, first_batch)

    print("\n--- Running Training Loop on Dummy Data ---")
    training_results = await miner.inner_steps(loader=dummy_loader, step_window=0)
    
    print("\n--- Dummy Training Loop Finished ---")
    print("Metrics returned by the loop:")
    print(json.dumps(training_results, indent=2))

def print_io_structure(miner, batch):
    """Prints the input/output structure for one forward pass."""
    # Unpack the tensor from the list
    input_ids = batch[0].to(miner.device)
    labels = input_ids.clone()

    print(f"\n--- Model Input Structure ---")
    print(f"  - `input_ids` (torch.Tensor): Shape={input_ids.shape}, DType={input_ids.dtype}")
    print(f"  - `labels` (torch.Tensor):    Shape={labels.shape}, DType={labels.dtype}")

    with torch.no_grad():
        outputs = miner.model(input_ids=input_ids, labels=labels)

    print("\n--- Model Output Structure ---")
    print(f"  - `outputs` is a {type(outputs).__name__}")
    print(f"  - `outputs.loss` (torch.Tensor): Shape={outputs.loss.shape}, Value={outputs.loss.item():.4f}")
    print(f"  - `outputs.logits` (torch.Tensor): Shape={outputs.logits.shape}, DType={outputs.logits.dtype}")

if __name__ == "__main__":
    asyncio.run(main())
