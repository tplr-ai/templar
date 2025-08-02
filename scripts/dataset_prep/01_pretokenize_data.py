import argparse
import glob
import math
import multiprocessing as mp
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


# Global tokenizer for multiprocessing
_tokenizer = None


def init_worker(tokenizer_name):
    """Initialize tokenizer in each worker process."""
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    _tokenizer.model_max_length = int(1e9)
    if _tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer {tokenizer_name} must have an EOS token.")


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


def main(args):
    target_tokens = args.total_tokens
    shard_size = args.shard_size
    expected_shards = math.ceil(target_tokens / shard_size)

    # Check existing shards
    if os.path.isdir(args.output_dir):
        existing = glob.glob(os.path.join(args.output_dir, "train_*.npy"))
        existing_count = sum(1 for f in existing if os.path.basename(f)[6:-4].isdigit())

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

    num_proc = args.num_proc if args.num_proc > 0 else max(1, os.cpu_count() * 3 // 4)
    print(f"Using {num_proc} processes")

    # Initialize processing variables
    shard_idx = 0
    shard_buffer = np.empty(shard_size, dtype=np.uint16)
    tokens_in_shard = 0
    total_tokens = 0

    with mp.Pool(num_proc, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
        with tqdm(total=target_tokens, unit="tokens", desc="Tokenizing") as pbar:
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
                        shard_path = os.path.join(
                            args.output_dir, f"train_{shard_idx:06d}.npy"
                        )
                        np.save(shard_path, shard_buffer)
                        tqdm.write(
                            f"Saved shard {shard_idx} ({shard_size / 1e6:.0f}M tokens)"
                        )
                        shard_idx += 1
                        tokens_in_shard = 0

    # Save final partial shard
    if tokens_in_shard > 0:
        shard_path = os.path.join(args.output_dir, f"train_{shard_idx:06d}.npy")
        np.save(shard_path, shard_buffer[:tokens_in_shard])
        tqdm.write(
            f"Saved final shard {shard_idx} ({tokens_in_shard / 1e6:.1f}M tokens)"
        )

    print(f"Completed. Total tokens: {total_tokens:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-tokenize dataset and save as shards"
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
        help="Output root directory for shards",
    )

    parser.add_argument(
        "--shard_size",
        type=int,
        default=1 * (1024**3),
        help="Tokens per shard (default: 1G)",
    )
    parser.add_argument(
        "--total_tokens",
        type=int,
        default=150 * (1024**3),
        help="Total tokens to process (default: 150G)",
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
    args.output_dir = os.path.expanduser(args.output_dir)

    main(args)
