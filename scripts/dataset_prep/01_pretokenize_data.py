import os
import argparse
import multiprocessing as mp
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import math
import glob
import boto3
import io

# Global tokenizer for multiprocessing
_tokenizer = None
_s3_client = None


def init_worker(tokenizer_name, r2_args=None):
    """Initialize tokenizer and S3 client in each worker process."""
    global _tokenizer, _s3_client
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    _tokenizer.model_max_length = int(1e9)
    if _tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer {tokenizer_name} must have an EOS token.")

    if r2_args and r2_args["bucket"]:
        _s3_client = boto3.client(
            "s3",
            endpoint_url=r2_args["endpoint_url"],
            aws_access_key_id=r2_args["access_key_id"],
            aws_secret_access_key=r2_args["secret_access_key"],
        )


def tokenize_doc(doc):
    """Tokenize a single document and append EOS token."""
    global _tokenizer
    text = doc.get("text")

    if not text or not isinstance(text, str):
        return np.array([], dtype=np.uint16)

    tokens = _tokenizer.encode(text, add_special_tokens=False)
    tokens.append(_tokenizer.eos_token_id)

    tokens_array = np.array(tokens, dtype=np.uint16)

    if not ((0 <= tokens_array) & (tokens_array < 2**16)).all():
        raise ValueError(
            f"Token IDs exceed uint16 range. Vocab size: {_tokenizer.vocab_size}"
        )

    return tokens_array


def save_shard(shard_buffer, shard_idx, tokens_in_shard, args):
    """Save a shard to local disk or R2."""
    shard_name = f"train_{shard_idx:06d}.npy"
    if args.r2_bucket:
        global _s3_client
        if _s3_client is None:
            _s3_client = boto3.client(
                "s3",
                endpoint_url=args.r2_endpoint_url,
                aws_access_key_id=args.r2_access_key_id,
                aws_secret_access_key=args.r2_secret_access_key,
            )

        buffer = io.BytesIO()
        np.save(buffer, shard_buffer[:tokens_in_shard])
        buffer.seek(0)

        key = os.path.join(args.r2_prefix, shard_name)
        _s3_client.upload_fileobj(buffer, args.r2_bucket, key)
        tqdm.write(
            f"Uploaded shard {shard_idx} to R2: s3://{args.r2_bucket}/{key} ({tokens_in_shard / 1e6:.1f}M tokens)"
        )
    else:
        shard_path = os.path.join(args.output_dir, shard_name)
        np.save(shard_path, shard_buffer[:tokens_in_shard])
        tqdm.write(
            f"Saved shard {shard_idx} to {shard_path} ({tokens_in_shard / 1e6:.1f}M tokens)"
        )


def main(args):
    target_tokens = args.total_tokens
    shard_size = args.shard_size
    expected_shards = math.ceil(target_tokens / shard_size)

    s3_client = None
    existing_count = 0
    if args.r2_bucket:
        print("R2 mode enabled. Will upload to R2 bucket.")
        s3_client = boto3.client(
            "s3",
            endpoint_url=f"https://{args.r2_endpoint_url}.r2.cloudflarestorage.com",
            aws_access_key_id=args.r2_access_key_id,
            aws_secret_access_key=args.r2_secret_access_key,
            region_name="auto",
        )

        # Check existing shards in R2
        try:
            response = s3_client.list_objects_v2(
                Bucket=args.r2_bucket, Prefix=os.path.join(args.r2_prefix, "train_")
            )
            existing_count = len(response.get("Contents", []))
        except Exception as e:
            print(f"Could not list objects in R2 bucket: {e}")
            existing_count = 0

        if existing_count >= expected_shards:
            print(
                f"Found {existing_count} shards in R2 (>= {expected_shards} expected). Skipping."
            )
            return
        else:
            print(f"Found {existing_count}/{expected_shards} shards in R2. Continuing.")

    else:
        # Check existing shards locally
        print(f"Running Locally")
        if os.path.isdir(args.output_dir):
            existing = glob.glob(os.path.join(args.output_dir, "train_*.npy"))
            existing_count = sum(
                1 for f in existing if os.path.basename(f)[6:-4].isdigit()
            )

            if existing_count >= expected_shards:
                print(
                    f"Found {existing_count} shards (>= {expected_shards} expected). Skipping."
                )
                return
            else:
                print(f"Found {existing_count}/{expected_shards} shards. Continuing.")
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Target: {target_tokens / 1e9:.1f}B tokens")
    print(f"Shard size: {shard_size / 1e6:.0f}M tokens")
    print(f"Expected shards: {expected_shards}")

    # Load and shuffle dataset
    dataset = load_dataset(
        args.dataset, split="train", streaming=True, trust_remote_code=True
    )
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    num_proc = args.num_proc if args.num_proc > 0 else min(os.cpu_count() - 1, os.cpu_count() * 9 // 10)
    print(f"Using {num_proc} processes")

    # Initialize processing variables
    shard_idx = existing_count
    shard_buffer = np.empty(shard_size, dtype=np.uint16)
    tokens_in_shard = 0
    total_tokens = shard_idx * shard_size

    r2_init_args = {
        "bucket": args.r2_bucket,
        "endpoint_url": args.r2_endpoint_url,
        "access_key_id": args.r2_access_key_id,
        "secret_access_key": args.r2_secret_access_key,
    }

    r2_init_args = {k: v for k, v in r2_init_args.items() if v} or None

    with mp.Pool(
        num_proc, initializer=init_worker, initargs=(args.tokenizer, r2_init_args)
    ) as pool:
        with tqdm(
            total=target_tokens, initial=total_tokens, unit="tokens", desc="Tokenizing"
        ) as pbar:
            for doc_tokens in pool.imap(
                tokenize_doc, iter(dataset), chunksize=args.chunk_size
            ):
                if total_tokens >= target_tokens:
                    break

                if len(doc_tokens) == 0:
                    continue

                # Process tokens from this document
                doc_idx = 0
                while doc_idx < len(doc_tokens) and total_tokens < target_tokens:
                    space_left = shard_size - tokens_in_shard
                    doc_left = len(doc_tokens) - doc_idx
                    global_left = target_tokens - total_tokens

                    take = min(space_left, doc_left, global_left)
                    if take == 0:
                        break

                    # Copy tokens to shard buffer
                    shard_buffer[tokens_in_shard : tokens_in_shard + take] = doc_tokens[
                        doc_idx : doc_idx + take
                    ]
                    tokens_in_shard += take
                    total_tokens += take
                    pbar.update(take)
                    doc_idx += take

                    # Save shard if full
                    if tokens_in_shard == shard_size:
                        save_shard(shard_buffer, shard_idx, tokens_in_shard, args)
                        shard_idx += 1
                        tokens_in_shard = 0

    # Save final partial shard
    if tokens_in_shard > 0:
        save_shard(shard_buffer, shard_idx, tokens_in_shard, args)

    print(f"Completed. Total tokens: {total_tokens:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-tokenize dataset and save as shards locally or to R2"
    )

    parser.add_argument(
        "--dataset",
        default="mlfoundations/dclm-baseline-1.0-parquet",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--tokenizer",
        default="togethercomputer/LLaMA-2-7B-32K",
        help="Transformers tokenizer",
    )
    parser.add_argument(
        "--output_dir",
        default="~/datasets/dclm_150B_tokenized",
        help="Output root directory for shards (if not using R2)",
    )

    # R2 arguments
    parser.add_argument(
        "--r2_bucket",
        default=os.getenv("R2_DATASET_BUCKET_NAME"),
        help="R2 bucket name",
    )
    parser.add_argument(
        "--r2_prefix",
        default="remote/tokenized/",
        help="R2 prefix for shards",
    )
    parser.add_argument(
        "--r2_endpoint_url",
        default=os.getenv("R2_DATASET_ACCOUNT_ID"),
        help="R2 endpoint URL",
    )
    parser.add_argument(
        "--r2_access_key_id",
        default=os.getenv("R2_DATASET_WRITE_ACCESS_KEY_ID"),
        help="R2 access key ID",
    )
    parser.add_argument(
        "--r2_secret_access_key",
        default=os.getenv("R2_DATASET_WRITE_SECRET_ACCESS_KEY"),
        help="R2 secret access key",
    )

    parser.add_argument(
        "--shard_size",
        type=int,
        default=100 * (1024**3),  # 100G tokens
        help="Tokens per shard (default: 100G)",
    )
    parser.add_argument(
        "--total_tokens",
        type=int,
        default=1000 * (1024**3),  # 1T tokens
        help="Total tokens to process (default: 1T)",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for dataset shuffling"
    )
    parser.add_argument(
        "--buffer_size", type=int, default=10000, help="Shuffle buffer size"
    )

    parser.add_argument(
        "--num_proc", type=int, default=-1, help="Number of processes (-1 for auto)"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=256, help="Processing chunk size"
    )

    args = parser.parse_args()
    if not args.r2_bucket:
        args.output_dir = os.path.expanduser(args.output_dir)
    elif not all(
        [args.r2_endpoint_url, args.r2_access_key_id, args.r2_secret_access_key]
    ):
        raise ValueError(
            "R2 bucket specified, but endpoint, access key, or secret key is missing."
        )

    main(args)
