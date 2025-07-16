import os
import argparse
import multiprocessing as mp
import cytoolz as c
import cytoolz.curried as cc
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
import math
import glob

from dotenv import load_dotenv
load_dotenv()


from tplr import logger 


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


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

    text = doc["text"] # fail fast due to filter checks 

    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens.append(tokenizer.eos_token_id)



    # tokens_array = np.array(tokens, dtype=dtype) 

    # if not ((0 <= tokens_array) & (tokens_array < 2**16)).all(): # can do this with a filter over a chunk?
    #     raise ValueError(
    #         f"Token IDs exceed uint16 range. Vocab size: {tokenizer.vocab_size}"
    #     )

    return tokens # tokens_array


def main(args):
    debug = args.debug
    target_tokens = args.total_tokens
    shard_size = args.shard_size
    expected_shards = math.ceil(target_tokens / shard_size)

    os.makedirs(args.output_dir, exist_ok=True)

    # Check existing shards
    existing = glob.glob(os.path.join(args.output_dir, "train_*.npy"))

    # could we do this as a function / class+attribute so that if our indices change we can still check?
    number_existing_shards = sum(1 for f in existing if os.path.basename(f)[6:-4].isdigit())

    if number_existing_shards >= expected_shards:
        logger.info(
            f"Found {number_existing_shards} shards (>= {expected_shards} expected). Skipping."
        )
        return
    else:
        logger.info(f"Found {number_existing_shards}/{expected_shards} shards. Continuing.")


    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Tokenizer: {args.tokenizer}")
    logger.info(f"Target: {target_tokens / 1e9:.1f}B tokens")
    logger.info(f"Shard size: {shard_size / 1e6:.0f}M tokens")
    logger.info(f"Expected shards: {expected_shards}")

    # # Load and shuffle dataset
    dataset = load_dataset(
        args.dataset, split="train", streaming=True, trust_remote_code=True
    )
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    tokenizer = instantiate_tokenizer(args.tokenizer)
    tokenize_fn = c.curry(tokenize_doc, tokenizer)  

    # would we ever want fewer workers? 1 is good only if the loop is misbehaving
    num_proc = 1 if debug else min(os.cpu_count() - 1, int(os.cpu_count() * .9))  
    logger.info(f"Using {num_proc} processes")

    # Initialize processing variables
    seq_len = 2048
    seqs_per_shard = 1024 # shard_size // seq_len

    # was using args.chunk_size previously
    map_fn = cc.map if num_proc == 1 else get_pmap_fn(num_proc) 

    writer = c.pipe(
        tqdm(
            # (manual_dataset() for _ in range(10000000)), 
            dataset,
            total=204800,
            # total=target_tokens, 
            # unit="tokens", 
            desc="Tokenizing",
        ),  
        cc.take(204800),
        cc.partition_all(1), 
        cc.filter(lambda d: filter_dataset(d[0])),
        map_fn(
            cc.compose_left(
                c.first,
                tokenize_fn,
                iter,
            ),
        ),
        cc.concat,
        cc.partition_all(2048), # seq_len number of tokens
        cc.map(arrayify_list),
        cc.partition_all(1024), # seqs_per_shard
        enumerate,
        cc.map(c.curry(write_shards, args, logger)),
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


def arrayify_list(l: list, dtype: np.dtype = np.uint16) -> np.ndarray:
    return np.array(l, dtype=dtype)


def write_shards(args, logger, enumerated_tokens):
    shard_idx, tokens = enumerated_tokens
    tokens = np.concat(tokens)

    shard_path = os.path.join(
        args.output_dir, f"train_{shard_idx:06d}.npy"
    )
    np.save(shard_path, tokens)
    logger.info(f"Saved shard {shard_idx} ({len(tokens) / 1e6:.0f}M tokens)")
    return


def break_list(l: list):
    for i in l:
        yield i


def manual_dataset():
    d = {'text': 'Gulf of Ob\n\nGulf of Ob, Russian Obskaya Guba, large inlet of the Kara Sea indenting northwestern Siberia, between the peninsulas of Yamal and Gyda, in north-central Russia. The gulf forms the outlet for the Ob River, the delta of which is choked by a huge sandbar. The gulf is about 500 miles (800 km) in length and has a breadth varying between 20 and 60 miles (32 and 97 km). The depth of the sea at this point is 33–40 feet (10–12 m). Its eastern coastline is steep and rugged; the west is low-lying and marshy. Novy Port is the main port of the gulf.', 'url': 'https://www.britannica.com/print/article/423566', 'id': '<urn:uuid:b8f22bed-4bef-4da4-893a-3fc1afc83874>', 'language': 'en', 'language_score': 0.9554803371429443, 'fasttext_score': 0.20887362957000732}
    return d


def passthrough_print(x):
    print(type(x), x)
    return x



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
    parser.add_argument(
        "--debug", type=bool, default=False, help="use 1 proc for debugging"
    )

    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)

    main(args)
