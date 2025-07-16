import argparse
import glob
import math
import multiprocessing as mp
import requests
import os

import cytoolz as c
import cytoolz.curried as cc
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm
from transformers import AutoTokenizer

load_dotenv()

from tplr import logger


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def instantiate_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """Initialize tokenizer in each worker process."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = int(1e9)
    if tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer {tokenizer_name} must have an EOS token.")
    return tokenizer


def tokenize_doc(
    tokenizer: AutoTokenizer,
    doc: list[dict[str, str]],
) -> list:
    """Tokenize a single document and append EOS token."""

    text = doc["text"]  # fail fast due to filter checks

    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens.append(tokenizer.eos_token_id)

    # tokens_array = np.array(tokens, dtype=dtype)

    # if not ((0 <= tokens_array) & (tokens_array < 2**16)).all(): # can do this with a filter over a chunk?
    #     raise ValueError(
    #         f"Token IDs exceed uint16 range. Vocab size: {tokenizer.vocab_size}"
    #     )

    return tokens  # tokens_array


def main(args):
    debug = args.debug
    target_tokens = args.total_tokens
    shard_size = args.shard_size
    seq_len = args.seq_len

    seqs_per_shard = shard_size // seq_len
    expected_shards = math.ceil(target_tokens / shard_size)

    os.makedirs(args.output_dir, exist_ok=True)

    # Check existing shards
    existing = glob.glob(os.path.join(args.output_dir, "train_*.npy"))

    # could we do this as a function / class+attribute so that if our indices change we can still check?
    number_existing_shards = sum(
        1 for f in existing if os.path.basename(f)[6:-4].isdigit()
    )

    if number_existing_shards >= expected_shards:
        logger.info(
            f"Found {number_existing_shards} shards (>= {expected_shards} expected). Skipping."
        )
        return
    else:
        logger.info(
            f"Found {number_existing_shards}/{expected_shards} shards. Continuing."
        )

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Tokenizer: {args.tokenizer}")
    logger.info(f"Target: {target_tokens / 1e9:.1f}B tokens")
    logger.info(f"Shard size: {shard_size / 1e6:.0f}M tokens")
    logger.info(f"Expected shards: {expected_shards}")

    # Load and shuffle dataset
    dataset = load_dataset(
        args.dataset, split="train", streaming=True, trust_remote_code=True
    )
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.buffer_size)
    dataset_row_count = retrieve_hf_rowcount(args.dataset)

    # Get tokenizer and potentially modify dtype
    tokenizer = instantiate_tokenizer(args.tokenizer)
    max_token_id = len(tokenizer) - 1
    token_dtype = handle_token_dtype(max_token_id)

    # would we ever want fewer workers? 1 is good only if the loop is misbehaving
    num_proc = 1 if debug else min(os.cpu_count() - 1, int(os.cpu_count() * 0.9))
    logger.info(f"Using {num_proc} processes")

    # was using args.chunk_size previously
    map_fn = cc.map if num_proc == 1 else get_pmap_fn(num_proc)

    writer = c.pipe(
        tqdm(
            dataset,
            total=dataset_row_count,
            unit="documents",
            desc="Tokenizing",
        ),
        cc.partition_all(1),
        cc.filter(lambda d: filter_dataset(d[0])),
        map_fn(
            cc.compose_left(
                c.first,
                c.curry(tokenize_doc, tokenizer),
                iter,
            ),
        ),
        cc.concat,
        cc.partition_all(seq_len),
        cc.map(c.curry(np.array, dtype=token_dtype)),
        cc.partition_all(seqs_per_shard),
        enumerate,
        cc.map(c.curry(write_shards, args, logger)),
        c.curry(tqdm, total=expected_shards),
        cc.take(expected_shards),  # stop iterating when we have correct shards
        cc.filter(bool),  # remove None outputs for RAM safety
        list,
    )

    total_tokens = seq_len * seqs_per_shard

    logger.info(f"Completed. Total tokens: {total_tokens:,}")
    return


def get_pmap_fn(num_proc: int):
    return c.curry(
        mp.Pool(num_proc).imap,
        chunksize=1,
    )


def filter_dataset(d: dict) -> bool:
    has_length = False

    text = d.get("text", "")

    is_valid_object = bool(text)
    is_string = isinstance(text, str)
    if is_string:
        has_length = len(text) > 0

    return all([is_valid_object, is_string, has_length])


def write_shards(args, logger, enumerated_tokens):
    shard_idx, tokens = enumerated_tokens
    tokens = np.concat(tokens)

    shard_path = os.path.join(args.output_dir, f"train_{shard_idx:06d}.npy")
    np.save(shard_path, tokens)
    logger.info(f"Saved shard {shard_idx} ({len(tokens) / 1e6:.0f}M tokens)")
    return


def handle_token_dtype(max_token_id: int) -> np.dtype:
    token_dtype = np.uint16  # 65_535 max token value
    if max_token_id > 65_535:
        token_dtype = np.uint32
    return token_dtype


def retrieve_hf_rowcount(dataset_name: str) -> int:
    """Retrieve the row count of a HuggingFace dataset."""

    url = f"https://datasets-server.huggingface.co/size?dataset={dataset_name}"
    response = requests.get(url).json()

    if "size" not in response:
        total_rows = 1_000_000_000  # default to 1B if not found
        logger.info(
            f"Dataset {dataset_name} size not found or no size available. Defaulting to {total_rows:,} rows."
        )
    else:
        total_rows = response["size"]["dataset"]["num_rows"]
        logger.info(f"Total rows in {dataset_name}: {total_rows}")

    return total_rows


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
        "--seq_len",
        type=int,
        default=2048,
        help="Model context length for sequence chunking (default: 2048)",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for dataset shuffling"
    )

    parser.add_argument(
        "--buffer_size", type=int, default=10240, help="Shuffle buffer size"
    )

    parser.add_argument(
        "--debug", type=bool, default=False, help="use 1 proc for debugging"
    )

    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)

    main(args)
