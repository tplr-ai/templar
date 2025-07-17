#!/usr/bin/env python3

import os
import argparse
import multiprocessing as mp
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

_tokenizer = None


def init_worker(tokenizer_name):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    _tokenizer.model_max_length = int(1e9)
    if _tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer {tokenizer_name} must have an EOS token.")


def tokenize_doc(doc):
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
    shard_path = os.path.join(args.output_dir, "train_000000.npy")
    if os.path.exists(shard_path):
        existing_size = np.load(shard_path, mmap_mode="r").shape[0]
        print(f"Shard already exists with {existing_size:,} tokens at {shard_path}")
        if existing_size >= args.shard_size * 0.9:
            print("Shard is sufficiently large. Skipping generation.")
            return
        else:
            print("Existing shard is too small. Regenerating...")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Target: {args.shard_size / 1e6:.0f}M tokens ({args.shard_size / 1e9:.1f}B)")
    print(f"Output: {shard_path}")

    dataset = load_dataset(
        args.dataset, split="train", streaming=True, trust_remote_code=True
    ).shuffle(seed=args.seed, buffer_size=args.buffer_size)

    num_proc = args.num_proc if args.num_proc > 0 else max(1, os.cpu_count() * 3 // 4)
    print(f"Using {num_proc} processes")

    shard_buffer = np.empty(args.shard_size, dtype=np.uint16)
    tokens_in_shard = 0

    with mp.Pool(num_proc, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
        with tqdm(total=args.shard_size, unit="tokens", desc="Tokenizing") as pbar:
            for doc_tokens in pool.imap(
                tokenize_doc, iter(dataset), chunksize=args.chunk_size
            ):
                if tokens_in_shard >= args.shard_size:
                    break
                if len(doc_tokens) == 0:
                    continue

                doc_idx = 0
                while doc_idx < len(doc_tokens) and tokens_in_shard < args.shard_size:
                    space_left = args.shard_size - tokens_in_shard
                    doc_left = len(doc_tokens) - doc_idx
                    take = min(space_left, doc_left)

                    if take == 0:
                        break

                    shard_buffer[tokens_in_shard : tokens_in_shard + take] = doc_tokens[
                        doc_idx : doc_idx + take
                    ]
                    tokens_in_shard += take
                    pbar.update(take)
                    doc_idx += take

    final_tokens = shard_buffer[:tokens_in_shard]
    np.save(shard_path, final_tokens)

    file_size_mb = os.path.getsize(shard_path) / (1024**2)
    print("\nSaved representative shard:")
    print(f"   Path: {shard_path}")
    print(f"   Tokens: {tokens_in_shard:,}")
    print(f"   Size: {file_size_mb:.1f} MB")
    print(f"   Dtype: {final_tokens.dtype}")

    print("\nQuick Stats:")
    print(f"   Vocab range: {final_tokens.min()} - {final_tokens.max()}")
    print(f"   Unique tokens: {len(np.unique(final_tokens)):,}")
    print(f"   Mean token ID: {final_tokens.mean():.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a single representative shard from the DCLM dataset"
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
        default="./test_shard",
        help="Output directory for the test shard",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=1 * (1024**3),
        help="Tokens in shard (default: 1B)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for dataset shuffling")
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
