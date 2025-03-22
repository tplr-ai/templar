#!/usr/bin/env python3
# checkpoint_manager.py - Bittensor model checkpoint and gradient application

import os
import re
import torch
import boto3
import argparse
import bittensor as bt
import time
from io import BytesIO
from transformers import LlamaConfig, LlamaForCausalLM
from dataclasses import dataclass
from typing import Optional, Tuple
from boto3.s3.transfer import TransferConfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.optim import SGD

# Configuration
BUCKET_SECRETS = {
    "gradients": {
        "account_id": "c71f5172df178f32158e87b8220c3675",
        "name": "c71f5172df178f32158e87b8220c3675",
        "access_key_id": "",
        "secret_key": "",
    }
}

HPARAMS = {
    "learning_rate": 4e-4,
    "weight_decay": 0.1,
}


@dataclass
class Bucket:
    """Simple bucket configuration class"""

    name: str
    account_id: str
    access_key_id: str
    secret_access_key: str


# Tensor operations
def unpack_binary_tensor(packed_tensor, original_shape):
    """Unpack a 1-bit representation tensor back to ±1 values."""
    total_elements = torch.prod(torch.tensor(original_shape)).item()

    # Create a flat tensor to hold the unpacked values
    unpacked = torch.zeros(total_elements, dtype=torch.float32)

    for i in range(8):
        mask = 1 << i
        bits = (packed_tensor & mask) >> i
        # Convert 0/1 to -1/+1
        unpacked[i::8] = (bits.float() * 2) - 1

    return unpacked.reshape(original_shape)


# S3/R2 storage utilities
def download_to_buffer(
    bucket_name,
    key,
    endpoint,
    access_key,
    secret_key,
    max_concurrency=10,
    chunk_size_mb=5,
):
    """Download file from R2/S3 in parallel chunks to an in-memory buffer."""
    # Configure the S3 client
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=boto3.session.Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=5,
            read_timeout=60,
            max_pool_connections=max_concurrency,
        ),
    )

    buffer = BytesIO()

    # TransferConfig for multipart download
    transfer_config = TransferConfig(
        multipart_threshold=chunk_size_mb * 1024 * 1024,  # start multipart at X MB
        multipart_chunksize=chunk_size_mb * 1024 * 1024,  # each chunk is X MB
        max_concurrency=max_concurrency,
        use_threads=True,
    )

    # Download in parallel chunks
    start = time.time()
    client.download_fileobj(
        Bucket=bucket_name, Key=key, Fileobj=buffer, Config=transfer_config
    )
    end = time.time()

    print(
        f"Downloaded {key} in {end - start:.2f}s using {max_concurrency} threads and {chunk_size_mb}MB chunk."
    )
    buffer.seek(0)
    return buffer


def s3_get_object(key: str, bucket: Bucket, timeout: int = 10) -> Optional[dict]:
    """Get an object from S3/R2 storage."""
    try:
        # Create S3 client
        s3_client = boto3.client(
            "s3",
            endpoint_url=f"https://{bucket.account_id}.r2.cloudflarestorage.com",
            region_name="auto",
            aws_access_key_id=bucket.access_key_id,
            aws_secret_access_key=bucket.secret_access_key,
        )

        # Get object head to get content length
        head_response = s3_client.head_object(Bucket=bucket.name, Key=key)
        total_size = head_response["ContentLength"]

        # Create a buffer to store the file
        buffer = BytesIO()

        # Progress tracking variables
        downloaded_bytes = 0
        last_percent_reported = 0

        def progress_callback(chunk_size):
            nonlocal downloaded_bytes, last_percent_reported
            downloaded_bytes += chunk_size
            percent = int(downloaded_bytes * 100 / total_size)
            if (
                percent >= last_percent_reported + 5 or percent == 100
            ):  # Report every 5%
                print(
                    f"Downloading {key}: {percent}% ({downloaded_bytes / 1024 / 1024:.2f}MB/{total_size / 1024 / 1024:.2f}MB)"
                )
                last_percent_reported = percent

        # Get object with progress tracking
        s3_client.download_fileobj(
            Bucket=bucket.name, Key=key, Fileobj=buffer, Callback=progress_callback
        )

        # Reset buffer position and load
        buffer.seek(0)
        print("Download complete. Loading checkpoint into memory...")
        loaded_data = torch.load(buffer, map_location="cpu")
        print("Checkpoint loaded into memory successfully.")

        return loaded_data

    except Exception as e:
        print(f"Error getting object {key}: {e}")
        return None


# Gradient loading functions
def load_aggregation(window, version="0.2.31", show_info=True):
    """Load aggregated gradients for a specified window using parallel-chunk S3/R2 download."""
    filename = f"aggregation-{window}-v{version}.pt"
    endpoint = (
        f"https://{BUCKET_SECRETS['gradients']['account_id']}.r2.cloudflarestorage.com"
    )

    # 1) Download to buffer using multipart
    buffer = download_to_buffer(
        bucket_name=BUCKET_SECRETS["gradients"]["name"],
        key=filename,
        endpoint=endpoint,
        access_key=BUCKET_SECRETS["gradients"]["access_key_id"],
        secret_key=BUCKET_SECRETS["gradients"]["secret_key"],
        max_concurrency=20,  # Try 20 or more if your CPU/network can handle it
        chunk_size_mb=8,  # Larger chunks reduce overhead
    )

    # 2) Load the compressed state from buffer
    compressed_data = torch.load(buffer, map_location="cpu")

    # 4) Process the data
    return process_loaded_data(compressed_data, show_info)


def process_loaded_data(compressed_data, show_info=True):
    """Unpack compressed 1-bit tensors."""
    # Create a dummy model just to get the shapes
    model_config = LlamaConfig(
        hidden_size=2048,
        num_hidden_layers=16,
        num_attention_heads=8,
        intermediate_size=8192,
        num_key_value_heads=8,
        max_position_embeddings=2048,
    )
    model = LlamaForCausalLM(model_config)

    result = {
        "timestamp": compressed_data.get("timestamp", None),
        "window": compressed_data.get("window", None),
        "version": compressed_data.get("version", None),
        "tensors": {},
    }

    for name, param in model.named_parameters():
        if name in compressed_data:
            original_shape = param.shape
            unpacked = unpack_binary_tensor(compressed_data[name], original_shape)
            result["tensors"][name] = unpacked

            if show_info:
                print(f"Unpacked tensor {name} with shape {original_shape}")

    if show_info:
        print(f"Successfully unpacked {len(result['tensors'])} tensors")

    return result


# Bittensor network interaction
def get_highest_stake_validator_bucket(
    subtensor, netuid: int
) -> Tuple[Optional[Bucket], Optional[int]]:
    """Get the bucket for the validator with highest stake."""
    # Get metagraph
    metagraph = subtensor.metagraph(netuid)

    # Get validator with highest stake
    validator_uid = metagraph.S.argmax().item()
    print(f"Found validator with highest stake: {validator_uid}")

    if validator_uid is None:
        print("No active validators found")
        return None, None

    # Get commitment only for the highest stake validator
    try:
        commitment = subtensor.get_commitment(netuid=netuid, uid=int(validator_uid))
        if commitment:
            validator_bucket = Bucket(
                name=commitment[:32].strip(),
                account_id=commitment[:32].strip(),
                access_key_id=commitment[32:64].strip(),
                secret_access_key=commitment[64:].strip(),
            )
            print(
                f"Retrieved bucket commitment for highest stake validator UID {validator_uid}"
            )
            return validator_bucket, validator_uid
        else:
            print(
                f"No commitment found for highest stake validator UID {validator_uid}"
            )
            return None, None
    except Exception as e:
        print(
            f"Failed to get commitment for highest stake validator UID {validator_uid}: {e}"
        )
        return None, None


def get_bucket_checkpoint(
    bucket: Bucket, uid: int, version: str = "0.2.28"
) -> Optional[Tuple[dict, int]]:
    """Helper to get checkpoint from a specific bucket."""
    try:
        # Create S3 client
        s3_client = boto3.client(
            "s3",
            endpoint_url=f"https://{bucket.account_id}.r2.cloudflarestorage.com",
            region_name="auto",
            aws_access_key_id=bucket.access_key_id,
            aws_secret_access_key=bucket.secret_access_key,
        )

        # Pattern to match checkpoint files
        pattern = re.compile(rf"^checkpoint-(\d+)-{uid}-v{version}\.pt$")

        # List objects in bucket
        response = s3_client.list_objects_v2(
            Bucket=bucket.name, Prefix="checkpoint", MaxKeys=1000
        )

        if not response.get("Contents"):
            return None

        # Find valid checkpoints
        valid_checkpoints = []
        for obj in response.get("Contents", []):
            key = obj.get("Key", "")
            match = pattern.match(key)
            if match:
                valid_checkpoints.append(
                    {
                        "key": key,
                        "window": int(match.group(1)),
                        "last_modified": obj["LastModified"],
                    }
                )

        # Get latest checkpoint
        if valid_checkpoints:
            latest = max(valid_checkpoints, key=lambda x: int(x["window"]))
            loaded_data = s3_get_object(key=latest["key"], bucket=bucket)
            if loaded_data:
                return loaded_data, latest["window"]

        return None

    except Exception as e:
        print(f"Error getting bucket checkpoint: {e}")
        return None


def get_debug_dict(
    bucket: Bucket, uid: int, window: int, version: str = "0.2.31"
) -> Optional[dict]:
    """Get debug dictionary from validator bucket for a specific window."""
    try:
        # The key for debug data
        key = f"debug-{window}-{uid}-v{version}.pt"

        print(
            f"Attempting to retrieve debug dictionary for window {window} from validator {uid}"
        )

        # Get the debug dictionary
        debug_data = s3_get_object(key=key, bucket=bucket)

        if debug_data:
            print(f"Successfully retrieved debug dictionary for window {window}")
            return debug_data
        else:
            # Try alternative key format if the first attempt failed
            key = f"debug-{window}-{uid}"
            debug_data = s3_get_object(key=key, bucket=bucket)

            if debug_data:
                print(
                    "Successfully retrieved debug dictionary using alternate key format"
                )
                return debug_data
            else:
                print(f"No debug dictionary found for window {window}")
                return None

    except Exception as e:
        print(f"Error getting debug dictionary: {e}")
        return None


# Model loading & checkpoint management
def load_checkpoint(
    model, netuid: int = 3, device: str = "cuda", version: str = "0.2.31"
) -> Tuple[bool, int, dict, int]:
    """Load the latest checkpoint from the highest stake validator."""
    try:
        # Initialize subtensor
        subtensor = bt.subtensor(network="finney")

        # Get highest stake validator bucket
        validator_bucket, validator_uid = get_highest_stake_validator_bucket(
            subtensor, netuid
        )

        if not validator_bucket or validator_uid is None:
            print("No validator bucket found")
            return False, 0, None, 0

        # Get checkpoint from validator bucket
        checkpoint_result = get_bucket_checkpoint(
            validator_bucket, validator_uid, version
        )

        if not checkpoint_result:
            print("No checkpoint found in validator bucket")
            return False, 0, None, 0

        checkpoint_data, window = checkpoint_result

        # Load model state dict
        if "model_state_dict" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model_state_dict"])
            model.to(device)
            print(f"Successfully loaded model from checkpoint at window {window}")

            # Get global step
            global_step = checkpoint_data.get("global_step", 0)
            if global_step == 0 and "start_window" in checkpoint_data:
                # Calculate global step from window and start window
                start_window = checkpoint_data.get("start_window", 0)
                current_window = checkpoint_data.get("current_window", window)
                global_step = current_window - start_window

            print(f"Global step: {global_step}")
            return True, global_step, checkpoint_data, window
        else:
            print("Invalid checkpoint data: missing model_state_dict")
            return False, 0, None, 0

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False, 0, None, 0


# Learning rate scheduling
def get_lr_at_step(global_step):
    """Get the learning rate at a specific global step."""
    # Create temporary objects to avoid modifying originals
    temp_optimizer = SGD([torch.tensor([1.0])], lr=HPARAMS["learning_rate"])

    # Recreate the schedulers
    temp_warmup_scheduler = LinearLR(
        temp_optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=250,
    )
    temp_cosine_scheduler = CosineAnnealingWarmRestarts(
        temp_optimizer,
        T_0=10000,
        T_mult=2,
        eta_min=HPARAMS["learning_rate"] * 0.1,
    )
    temp_scheduler = SequentialLR(
        temp_optimizer,
        schedulers=[temp_warmup_scheduler, temp_cosine_scheduler],
        milestones=[250],
    )

    # Step to the target global step
    for _ in range(global_step):
        temp_scheduler.step()

    # Return the current learning rate
    return temp_optimizer.param_groups[0]["lr"]


# Model validation
def calculate_model_debug_difference(model, debug_dict):
    """Calculate L2 norm of difference between model and debug values."""
    if not debug_dict or "state_dict" not in debug_dict:
        return None, None

    debug_state_dict = debug_dict["state_dict"]
    total_squared_diff = 0.0
    param_count = 0

    for name, param in model.named_parameters():
        # Check if there's a corresponding debug entry
        debug_key = name + "_debug"
        if debug_key in debug_state_dict:
            # Calculate L2 norm for this parameter
            param_data = param.data.cpu().flatten()[:2]  # Take only first two values
            debug_data = torch.tensor(debug_state_dict[debug_key]).cpu()
            squared_diff = torch.sum((param_data - debug_data) ** 2).item()
            total_squared_diff += squared_diff
            param_count += param_data.numel()

    # Final L2 norm across all parameters
    final_l2_norm = torch.sqrt(torch.tensor(total_squared_diff)).item()
    avg_norm = final_l2_norm / param_count if param_count > 0 else 0

    return final_l2_norm, avg_norm


def main(args):
    """Main function to load and apply checkpoints/gradients."""
    print(f"Starting checkpoint manager with version {args.version}")

    # Create model config
    model_config = LlamaConfig(
        hidden_size=2048,
        num_hidden_layers=16,
        num_attention_heads=8,
        intermediate_size=8192,
        num_key_value_heads=8,
        max_position_embeddings=2048,
    )

    # Create model
    model = LlamaForCausalLM(model_config)

    # Load checkpoint
    success, global_step, checkpoint_data, window = load_checkpoint(
        model=model,
        netuid=args.netuid,
        device=args.device,
        version=args.version,
    )

    if not success:
        print("Failed to load checkpoint, aborting.")
        return

    print(f"Checkpoint window: {window}")

    # Move model to device
    model = model.to(args.device)

    # Initialize subtensor to get validator bucket
    subtensor = bt.subtensor(network="finney")
    validator_bucket, validator_uid = get_highest_stake_validator_bucket(
        subtensor, args.netuid
    )

    if not validator_bucket or validator_uid is None:
        print("Unable to get validator bucket - cannot proceed with aggregation steps")
        return

    # Apply aggregation for each step
    for step in range(args.num_steps):
        step_window = window + step
        print(f"\nProcessing window {step_window}")

        # Load aggregation for current window
        agg_data = load_aggregation(
            window=step_window, version=args.version, show_info=True
        )
        if not agg_data:
            print(f"No aggregation data found for window {step_window}")
            continue

        # Get learning rate for this step
        lr = get_lr_at_step(global_step=step_window)
        print(f"Using learning rate: {lr:.6f}")

        # Get debug dictionary for current window
        # Note: there is a bug where debug_dict is saved with wrong window number and delayed by 1 or 2
        debug_dict = get_debug_dict(
            validator_bucket, validator_uid, step_window - 2, version=args.version
        )

        # Apply aggregation to model
        print("Applying aggregation to model...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in agg_data["tensors"]:
                    # Move aggregation tensor to device
                    agg_tensor = agg_data["tensors"][name].to(args.device)

                    # Apply weight decay to parameter
                    param.data.mul_(1.0 - lr * HPARAMS["weight_decay"])

                    # Apply gradient update with learning rate
                    param.data.add_(agg_tensor, alpha=-lr)

        print(f"Successfully applied aggregation for window {step_window}")

        # Calculate L2 norm of difference between model and debug values after this step
        if debug_dict:
            l2_norm, avg_norm = calculate_model_debug_difference(model, debug_dict)
            if l2_norm is not None:
                print(
                    f"Window {step_window} - L2 norm difference: {l2_norm:.6f}, Avg per param: {avg_norm:.6f}"
                )
        else:
            print(
                f"No valid debug dictionary available for window {step_window - 2} - cannot compute L2 norm"
            )

    # Save final model if requested
    if args.save_path:
        save_path = args.save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "window": window + args.num_steps,
                "global_step": global_step + args.num_steps,
                "applied_steps": args.num_steps,
                "version": args.version,
            },
            save_path,
        )
        print(
            f"Updated model with {args.num_steps} aggregation steps saved to {save_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bittensor Checkpoint Manager")
    parser.add_argument("--netuid", type=int, default=3, help="Network UID")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument("--version", type=str, default="0.2.35", help="Version string")
    parser.add_argument(
        "--num_steps", type=int, default=5, help="Number of steps to apply"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./model_updated.pt",
        help="Path to save updated model",
    )

    args = parser.parse_args()
    main(args)
