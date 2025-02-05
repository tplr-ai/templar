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
# fmt: off

import os
import json
import shutil
import torch
import asyncio
import argparse
import bittensor as bt
from transformers import LlamaForCausalLM
import templar as tplr
import wandb


class Evaluator:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description="Evaluator script")
        parser.add_argument(
            "--project", type=str, default="templar", help="Optional wandb project name"
        )
        parser.add_argument(
            "--netuid", type=int, default=3, help="Bittensor network UID."
        )
        parser.add_argument(
            "--actual_batch_size", type=int, default=8, help="Evaluation batch size."
        )
        parser.add_argument(
            "--device", type=str, default="cuda", help="Device to use for evaluation"
        )
        parser.add_argument(
            "--tasks",
            type=str,
            default="arc_challenge,arc_easy,openbookqa,winogrande,piqa,hellaswag",
            help="Comma-separated list of tasks to evaluate",
        )
        parser.add_argument(
            "--checkpoint_path",
            type=str,
            default=None,
            help="Path to save/load checkpoints. Defaults to checkpoints/eval/",
        )
        parser.add_argument(
            "--eval_interval",
            type=int,
            default=500,
            help="Number of global steps between evaluations",
        )
        bt.subtensor.add_args(parser)
        config = bt.config(parser)
        return config

    def __init__(self):
        self.config = Evaluator.config()

        # Init bittensor objects
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)

        # Set checkpoint path
        if self.config.checkpoint_path is None:
            self.checkpoint_path = "checkpoints/"
        else:
            self.checkpoint_path = self.config.checkpoint_path
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        # Init model and hyperparameters
        self.hparams = tplr.load_hparams()
        self.model = LlamaForCausalLM(config=self.hparams.model_config)
        self.model.to(self.config.device)

        # Initialize wandb with proper group and job type
        self.wandb_run = tplr.initialize_wandb(
            run_prefix="E",
            uid=0,
            config=self.config,
            group="eval",
            job_type="evaluation",
        )

        # Get all buckets
        self.buckets = tplr.get_all_buckets(
            subtensor=self.subtensor,
            netuid=self.config.netuid,
            metagraph=self.metagraph,
        )

        # Track last evaluation global step
        self.last_eval_step = 0
        self.stop_event = asyncio.Event()
        self.last_checkpoint_timestamp = 0
        self.last_block_number = 0

    async def update_state(self):
        """Updates metagraph and buckets"""
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.buckets = tplr.get_all_buckets(
            subtensor=self.subtensor,
            netuid=self.config.netuid,
            metagraph=self.metagraph,
        )

    async def evaluate_highest_stake_model(self) -> None:
        """Evaluates the model from the highest stake neuron"""
        try:
            await self.update_state()

            try:
                # Use the simplified loader
                global_step, block_number = await tplr.load_model_for_eval(
                    metagraph=self.metagraph,
                    buckets=self.buckets,
                    model=self.model,
                    checkpoint_path=self.checkpoint_path,
                    device=self.config.device,
                )

                if global_step == 0:
                    tplr.logger.error(
                        "Failed to load checkpoint from highest stake neuron"
                    )
                    return None

                # Check if this is a new checkpoint based on block number
                if block_number <= self.last_block_number:
                    tplr.logger.info(
                        f"No new checkpoint available (current block: {block_number}, last evaluated: {self.last_block_number}). Skipping evaluation."
                    )
                    return global_step

                # Check if we should evaluate based on global step
                if global_step - self.last_eval_step < self.config.eval_interval:
                    tplr.logger.info(
                        f"Skipping evaluation: Only {global_step - self.last_eval_step} steps since last evaluation"
                    )
                    return global_step

                tplr.logger.info(
                    f"Starting evaluation at global step {global_step} (block {block_number})"
                )

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
                    f"--model_args pretrained={model_path},tokenizer={model_path} "
                    f"--tasks {self.config.tasks} "
                    f"--device {self.config.device} "
                    f"--batch_size {self.config.actual_batch_size} "
                    f"--output_path {results_dir}"
                )

                exit_code = os.system(lm_eval_command)
                if exit_code != 0:
                    tplr.logger.error("Evaluation failed")
                    return global_step

                # Process and log results
                eval_results_dir = os.path.join(results_dir, "models__eval")
                if not os.path.exists(eval_results_dir):
                    tplr.logger.error(
                        f"Results directory not found: {eval_results_dir}"
                    )
                    return global_step

                latest_file = max(
                    [
                        os.path.join(eval_results_dir, f)
                        for f in os.listdir(eval_results_dir)
                    ],
                    key=os.path.getctime,
                )

                try:
                    with open(latest_file, "r") as f:
                        results = json.load(f)

                    # Prepare metrics dictionary
                    metrics = {}
                    for task_name, task_results in results["results"].items():
                        metric_name = (
                            "acc_norm,none" if task_name != "winogrande" else "acc,none"
                        )
                        if metric_value := task_results.get(metric_name):
                            tplr.logger.info(f"{task_name}: {metric_value}")
                            # Log each metric with global_step
                            self.wandb_run.log(
                                {
                                    f"eval/{task_name}": float(metric_value),
                                    "global_step": global_step,
                                }
                            )

                    # Debug logging
                    tplr.logger.info(f"Prepared metrics for logging: {metrics}")
                    tplr.logger.info(f"Current global step: {global_step}")

                    # Verify wandb state
                    if not self.wandb_run:
                        tplr.logger.error("WandB not initialized, reinitializing...")
                        self.wandb_run = tplr.initialize_wandb(
                            run_prefix="E",
                            uid=0,
                            config=self.config,
                            group="eval",
                            job_type="evaluation",
                        )

                    if not hasattr(self.wandb_run, "run") or not self.wandb_run.run:
                        tplr.logger.error("WandB run not active, reinitializing...")
                        self.wandb_run = tplr.initialize_wandb(
                            run_prefix="E",
                            uid=0,
                            config=self.config,
                            group="eval",
                            job_type="evaluation",
                        )

                    # Log metrics and update charts
                    if metrics:
                        # Log individual metrics
                        self.wandb_run.log(metrics, step=global_step)

                        # Create comparison table
                        comparison_data = []
                        for task_name, task_results in results["results"].items():
                            metric_name = (
                                "acc_norm,none"
                                if task_name != "winogrande"
                                else "acc,none"
                            )
                            if metric_value := task_results.get(metric_name):
                                comparison_data.append(
                                    [
                                        task_name,
                                        float(metric_value),
                                        int(global_step),  # Store as integer
                                    ]
                                )

                        comparison_table = wandb.Table(
                            columns=["Task", "Performance", "Step"],
                            data=comparison_data,
                        )

                        # Log table separately
                        self.wandb_run.log({"task_comparison": comparison_table})

                        tplr.logger.info(
                            f"Successfully logged metrics to wandb at step {global_step}"
                        )

                        # Force sync
                        try:
                            self.wandb_run.log_artifact(
                                latest_file, name=f"eval_results_{global_step}"
                            )
                            tplr.logger.info("Successfully synced results to wandb")
                        except Exception as e:
                            tplr.logger.error(f"Error syncing to wandb: {str(e)}")
                    else:
                        tplr.logger.error("No metrics to log!")

                except Exception as e:
                    tplr.logger.error(f"Error processing and logging results: {str(e)}")
                    tplr.logger.exception(e)  # Print full traceback

                # Cleanup
                shutil.rmtree(model_path)
                torch.cuda.empty_cache()

                # Update timestamps after successful evaluation
                self.last_eval_step = global_step
                self.last_block_number = block_number
                return global_step

            except Exception as e:
                tplr.logger.error(f"Error during evaluation: {str(e)}")
                return None

        except Exception as e:
            tplr.logger.error(f"Error during evaluation: {str(e)}")
            return None

    async def run(self):
        """Main run loop"""
        try:
            while not self.stop_event.is_set():
                # Check if new checkpoint exists
                await self.update_state()

                # Get latest checkpoint info without loading model
                latest_block = max(
                    bucket.block_number
                    for bucket in self.buckets
                    if bucket.block_number
                )

                if latest_block > self.last_block_number:
                    tplr.logger.info(
                        f"New checkpoint found at block {latest_block}, running evaluation..."
                    )
                    await self.evaluate_highest_stake_model()
                else:
                    tplr.logger.info(
                        f"No new checkpoint (current: {latest_block}, last: {self.last_block_number})"
                    )

                # Keep original 600 second (10 minute) sleep time
                await asyncio.sleep(600)

        except KeyboardInterrupt:
            tplr.logger.info("Evaluation interrupted by user")
            self.stop_event.set()
        except Exception as e:
            tplr.logger.error(f"Evaluation failed: {str(e)}")
        finally:
            if self.wandb_run:
                self.wandb_run.finish()

    def cleanup(self):
        """Cleanup resources"""
        self.stop_event.set()
        if self.wandb_run:
            self.wandb_run.finish()


if __name__ == "__main__":
    evaluator = Evaluator()
    try:
        asyncio.run(evaluator.run())
    except KeyboardInterrupt:
        tplr.logger.info("Shutting down evaluator...")
    finally:
        evaluator.cleanup()
