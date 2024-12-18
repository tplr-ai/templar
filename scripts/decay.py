import os
import shutil
import sys
import json
import torch
import random
import asyncio
import numpy as np
from typing import Optional
from transformers import LlamaForCausalLM
import bittensor as bt
import argparse
import torch.optim as optim

import templar as tplr


class DecayAgent:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description="Decay agent script")
        parser.add_argument(
            "--project", type=str, default="templar", help="Optional wandb project name"
        )
        parser.add_argument(
            "--netuid", type=int, default=3, help="Bittensor network UID."
        )
        parser.add_argument(
            "--actual_batch_size",
            type=int,
            default=8,
            help="Training batch size per accumulation.",
        )
        parser.add_argument(
            "--device", type=str, default="cuda", help="Device to use for training"
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        parser.add_argument("--trace", action="store_true", help="Enable trace logging")
        parser.add_argument("--test", action="store_true", help="Run on test network")
        parser.add_argument("--local", action="store_true", help="Run on local network")
        parser.add_argument(
            "--checkpoint_path",
            type=str,
            default=None,
            help="Path to save/load the checkpoint",
        )
        parser.add_argument(
            "--save-location",
            type=str,
            default=None,
            help="Directory to save/load slice files",
        )
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        config = bt.config(parser)
        if config.test:
            config.subtensor.network = "test"
            config.subtensor.chain_endpoint = "wss://test.finney.opentensor.ai:443/"
        elif config.local:
            config.subtensor.network = "local"
            config.subtensor.chain_endpoint = "ws://127.0.0.1:9944"
        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()
        return config

    def __init__(self):
        # Init config
        self.config = DecayAgent.config()
        tplr.logger.info("\n" + "-" * 40 + " Config " + "-" * 40)
        tplr.logger.info(self.config)

        # Init bittensor objects
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.chain_manager = tplr.chain.ChainManager(
            subtensor=self.subtensor, wallet=self.wallet, netuid=self.config.netuid
        )
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            tplr.logger.error(
                f"\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}[/bold]. You need to register first with: [blue]`btcli subnet register`[/blue]\n"
            )
            sys.exit()
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        tplr.logger.info("\n" + "-" * 40 + " Objects " + "-" * 40)
        tplr.logger.info(
            f"\nWallet: {self.wallet}\nSubtensor: {self.subtensor}\nMetagraph: {self.metagraph}\nUID: {self.uid}"
        )

        # Set up paths
        self.checkpoint_path = os.path.join(
            "checkpoints",
            "decay",
            f"neuron_checkpoint_{self.wallet.hotkey.ss58_address}",
        )
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        # Initialize wandb
        wandb_dir = os.path.join(os.getcwd(), "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        run_id_file = os.path.join(
            wandb_dir, f"wandb_run_id_D{self.uid}_{tplr.__version__}.txt"
        )

        if os.path.exists(run_id_file):
            with open(run_id_file, "r") as f:
                run_id = f.read().strip()
            tplr.logger.info(f"Resuming WandB run with id {run_id}")
        else:
            run_id = None
            tplr.logger.info("Starting new WandB run")

        self.wandb = tplr.initialize_wandb(
            run_prefix="D",
            uid=self.uid,
            config=self.config,
            group="decay",
            job_type="decay_training",
        )

        # Load model and configuration
        self.hparams = tplr.load_hparams()
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        self.model = LlamaForCausalLM(config=self.hparams.model_config)
        self.model.to(self.config.device)

        # Initialize optimizer with decay schedule
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.optimizer_beta1, self.hparams.optimizer_beta2),
            weight_decay=self.hparams.optimizer_weight_decay,
            foreach=True,
        )

        # Initialize decay scheduler
        self.scheduler = tplr.get_wsd_scheduler(
            optimizer=self.optimizer,
            num_warmup_steps=0,  # No warmup for decay
            num_stable_steps=0,  # No stable phase
            num_decay_steps=self.hparams.num_decay_steps,
        )

        # Initialize checkpoint manager
        self.checkpoint_manager = tplr.CheckpointManager(
            model=self.model,
            checkpoint_path=self.checkpoint_path,
            wallet=self.wallet,
            device=self.config.device,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        # Load initial checkpoint with optimizer state
        self.global_step = asyncio.run(
            self.checkpoint_manager.load_from_highest_stake(
                metagraph=self.metagraph,
                buckets=self.buckets,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                is_validator=False,
                hparams=self.hparams,
            )
        )

        # Get buckets for all neurons
        self.buckets = tplr.get_all_buckets(
            subtensor=self.subtensor,
            netuid=self.config.netuid,
            metagraph=self.metagraph,
        )

        # Initialize state
        self.last_eval_step = 0
        self.last_block_number = 0
        self.global_step = 0

    async def decay_and_evaluate(self) -> Optional[int]:
        """Performs decay training and evaluation"""
        try:
            if self.global_step == 0:
                tplr.logger.error("Failed to load checkpoint from highest stake neuron")
                return None

            # Start decay training
            tplr.logger.info(f"Starting decay training from step {self.global_step}")

            for step in range(self.hparams.num_decay_steps):
                # Download training data
                pages = await tplr.dataset.DatasetLoader.next_pages(
                    offset=self.block_number, n_pages=1, seed=self.uid
                )

                dataset = await tplr.dataset.DatasetLoader.create(
                    batch_size=self.config.actual_batch_size,
                    sequence_length=self.hparams.sequence_length,
                    pages_info=pages,
                    tokenizer=self.hparams.tokenizer,
                )

                # Training step
                self.model.train()
                total_loss = 0.0

                for batch in dataset:
                    input_ids = torch.tensor(batch, dtype=torch.long).to(
                        self.model.device
                    )
                    labels = input_ids.clone()
                    labels = torch.where(
                        labels == self.hparams.tokenizer.pad_token_id, -100, labels
                    )

                    with torch.amp.autocast(
                        device_type=self.model.device.type, dtype=torch.bfloat16
                    ):
                        outputs = self.model(input_ids=input_ids, labels=labels)

                    total_loss += outputs.loss.item()
                    outputs.loss.backward()

                if self.hparams.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.hparams.grad_clip
                    )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                current_lr = self.scheduler.get_last_lr()[0]
                step_loss = total_loss / len(dataset)

                # Log training metrics
                self.wandb.log(
                    {
                        "decay/loss": step_loss,
                        "decay/learning_rate": current_lr,
                        "decay/progress": step / self.hparams.num_decay_steps,
                        "global_step": self.global_step + step,
                    }
                )

                # Run evaluation periodically
                if (step + 1) % self.config.eval_interval == 0:
                    await self.evaluate(self.global_step + step)

                tplr.logger.info(
                    f"Decay step {step}/{self.hparams.num_decay_steps}, Loss: {step_loss:.4f}, LR: {current_lr:.2e}"
                )

            tplr.logger.info("Decay training completed")
            return self.global_step + self.hparams.num_decay_steps

        except Exception as e:
            tplr.logger.error(f"Error during decay training: {str(e)}")
            return None

    async def evaluate(self, global_step: int) -> None:
        """Runs evaluation on the current model state"""
        try:
            # Save model and tokenizer for evaluation
            model_path = "models/eval"
            os.makedirs(model_path, exist_ok=True)
            self.model.save_pretrained(model_path)
            self.hparams.tokenizer.save_pretrained(model_path)

            # Create results directory
            results_dir = f"{model_path}/results"
            os.makedirs(results_dir, exist_ok=True)

            # Run evaluation
            lm_eval_command = (
                f"lm-eval "
                f"--model hf "
                f"--model_args pretrained=models/eval,tokenizer=models/eval "
                f"--tasks arc_challenge,arc_easy,hellaswag,openbookqa,piqa,winogrande "
                f"--device {self.config.device} "
                f"--batch_size 6 "
                f"--output_path {results_dir}"
            )

            process = await asyncio.create_subprocess_shell(
                lm_eval_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                tplr.logger.error(f"Evaluation failed: {stderr.decode()}")
                return

            # Process and log results
            eval_results_dir = os.path.join(results_dir, "models__eval")
            if not os.path.exists(eval_results_dir):
                tplr.logger.error(f"Results directory not found: {eval_results_dir}")
                return

            latest_file = max(
                [
                    os.path.join(eval_results_dir, f)
                    for f in os.listdir(eval_results_dir)
                ],
                key=os.path.getctime,
            )

            with open(latest_file, "r") as f:
                results = json.load(f)

            # Log results to wandb
            for task_name, task_results in results["results"].items():
                metric_name = (
                    "acc_norm,none" if task_name != "winogrande" else "acc,none"
                )
                if metric_value := task_results.get(metric_name):
                    tplr.logger.info(f"{task_name}: {metric_value}")
                    self.wandb.log(
                        {
                            f"eval/{task_name}": metric_value,
                        },
                        step=global_step,
                    )

            # Cleanup
            shutil.rmtree(model_path)
            torch.cuda.empty_cache()

        except Exception as e:
            tplr.logger.error(f"Error during evaluation: {str(e)}")


async def main():
    decay_agent = DecayAgent()
    await decay_agent.decay_and_evaluate()


if __name__ == "__main__":
    asyncio.run(main())
