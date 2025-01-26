# neurons/analyzer.py

# Standard library imports
import os
import time
import asyncio
import argparse
import numpy as np

# Third-party imports
import torch
import boto3
from typing import List
from botocore.config import Config

# Local imports
import tplr
from tplr.config import BUCKET_SECRETS

# Set random seeds and CUDA settings for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Analyzer:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description="Analyzer script")
        parser.add_argument(
            "--device", default="cuda", type=str, help="Device to use for computation"
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument("--trace", action="store_true", help="Enable trace mode")
        parser.add_argument(
            "--use_wandb",
            action="store_true",
            help="Use Weights and Biases for logging",
        )
        parser.add_argument(
            "--project", type=str, default="templar", help="Wandb project name"
        )
        parser.add_argument(
            "--analysis_interval",
            type=int,
            default=60,
            help="Interval between analyses in seconds",
        )
        parser.add_argument(
            "--recent_windows",
            type=int,
            default=5,
            help="Number of recent windows to analyze",
        )
        config = parser.parse_args()
        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()
        return config

    def __init__(self):
        # Initialize configuration
        self.config = Analyzer.config()
        self.hparams = tplr.load_hparams()

        # Initialize WandB
        self.wandb = tplr.initialize_wandb(
            run_prefix="A",
            uid=0,  # Analyzer UID can be set to 0 or any identifier
            config=self.config,
            group="analyzer",
            job_type="analysis",
        )

        # Initialize R2 client using BUCKET_SECRETS from config.py
        self.bucket_info = BUCKET_SECRETS["gradients"]
        self.r2_endpoint = (
            f"https://{self.bucket_info['account_id']}.r2.cloudflarestorage.com"
        )
        self.bucket_name = self.bucket_info["name"]
        self.access_key_id = self.bucket_info["credentials"]["read"]["access_key_id"]  # type: ignore
        self.secret_access_key = self.bucket_info["credentials"]["read"][
            "secret_access_key"
        ]  # type: ignore

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.r2_endpoint,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(signature_version="s3v4", max_pool_connections=256),
        )

        # Initialize state parameters
        self.processed_files = set()
        self.version = tplr.__version__
        self.temp_dir = os.path.join("/tmp", "analyzer")
        os.makedirs(self.temp_dir, exist_ok=True)

    async def run(self):
        while True:
            try:
                await self.analyze_gradients()
            except Exception as e:
                tplr.logger.error(f"Error during analysis: {e}")
            await asyncio.sleep(self.config.analysis_interval)

    async def analyze_gradients(self):
        # List all gradient files in R2 storage under the 'gathers' prefix for the current version
        prefix = f"gathers/{self.version}/"
        gradient_keys = self.list_gradient_files(prefix)

        tplr.logger.info(f"Found {len(gradient_keys)} gradient files to analyze.")

        tasks = []
        for key in gradient_keys:
            if key in self.processed_files:
                continue  # Skip already processed files
            tasks.append(self.process_gradient_file(key))

        if tasks:
            await asyncio.gather(*tasks)

    def list_gradient_files(self, prefix: str) -> List[str]:
        paginator = self.s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
        gradient_keys = []
        for page in page_iterator:
            contents = page.get("Contents", [])
            for obj in contents:
                key = obj["Key"]  # type: ignore
                if key.endswith(".npz"):
                    gradient_keys.append(key)
        return gradient_keys

    async def process_gradient_file(self, key: str):
        try:
            # Download the gradient file from R2
            local_file = os.path.join(self.temp_dir, os.path.basename(key))
            await self.download_file(key, local_file)

            # Load the gradient data
            data = np.load(local_file, allow_pickle=True)
            state_dict = data["state_dict"].item()
            metadata = data["metadata"].item()

            # Extract UID and window from metadata
            uid = int(metadata.get("uid", -1))
            window = int(metadata.get("window", -1))
            global_step = int(metadata.get("global_step", -1))

            # Analyze the gradients
            metrics = self.analyze_state_dict(state_dict)
            metrics.update(
                {
                    "uid": uid,
                    "window": window,
                    "global_step": global_step,
                    "timestamp": time.time(),
                }
            )

            # Log metrics to WandB
            if self.wandb:
                self.wandb.log(metrics)
            tplr.logger.info(f"Analyzed gradients for UID {uid}, Window {window}")

            # Mark file as processed
            self.processed_files.add(key)

            # Clean up
            if os.path.exists(local_file):
                os.remove(local_file)

        except Exception as e:
            tplr.logger.error(f"Error processing gradient file {key}: {e}")

    async def download_file(self, key: str, local_file: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.s3_client.download_file, self.bucket_name, key, local_file
        )

    def analyze_state_dict(self, state_dict):
        # Collect all 'vals' tensors
        vals_tensors = []
        for key, value in state_dict.items():
            if key.endswith("vals"):
                if isinstance(value, torch.Tensor):
                    vals_tensors.append(value.detach().cpu().numpy())
                elif isinstance(value, np.ndarray):
                    vals_tensors.append(value)
                else:
                    tplr.logger.warning(f"Unexpected data type for key {key}")
        if not vals_tensors:
            tplr.logger.warning("No 'vals' tensors found in state_dict")
            return {}

        # Concatenate all 'vals' tensors
        all_vals = np.concatenate([v.flatten() for v in vals_tensors])

        # Compute metrics
        metrics = {
            "gradient_norm": float(np.linalg.norm(all_vals)),
            "gradient_mean": float(np.mean(all_vals)),
            "gradient_std": float(np.std(all_vals)),
            "gradient_min": float(np.min(all_vals)),
            "gradient_max": float(np.max(all_vals)),
            "gradient_skewness": float(scipy.stats.skew(all_vals)),
            "gradient_kurtosis": float(scipy.stats.kurtosis(all_vals)),
            "gradient_sparsity": float(np.mean(all_vals == 0)),
        }
        return metrics


def main():
    analyzer = Analyzer()
    asyncio.run(analyzer.run())


if __name__ == "__main__":
    import scipy.stats  # type: ignore

    main()
