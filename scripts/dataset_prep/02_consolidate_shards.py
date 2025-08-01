import argparse
import asyncio
import boto3
import gc
import hashlib
import os
import struct
import time
from pathlib import Path

import numpy as np

from neurons import miner
from tqdm.auto import tqdm
import io
import cytoolz as c
from joblib import Parallel, delayed

from tplr import sharded_dataset, comms, hparams

EXPECTED = {
    "tokens": {
        "sha256": "4c53d81b27b059e83b0cde7f97fc9877572cf5081eb0a2ebec9b0fc00e79a931",
        "elements": 161061273600,
    },
    "sample_ids": {
        "sha256": "95dcefad84cae66784a09e938d5136dcb32b326243b67389d7ca82fc3cce3f3f",
        "elements": 78643200,
    },
}


def tokens_handler(x):
    x = x.tobytes()
    x = hashlib.blake2b(x, digest_size=8).digest()
    x = struct.unpack("<Q", x)[0]
    return x


def passthrough_print(x):
    print(x)
    return x


async def run_preprocessing(args, seq_len: int = 2048, token_dtype: np.dtype = np.uint16): #data_root: str, seq_len: int, token_dtype: np.dtype) -> bool:
    """
    Consolidates .npy shards into single 'tokens.bin' and 'sample_ids.bin'.
    Also prints a SHA-256 digest and element count for sample_ids.bin.
    """

    # shards_dir = Path(data_root)
    # files = sorted(shards_dir.glob("train_*.npy"))
    # if not files:
    #     raise FileNotFoundError(f"No train_*.npy shards found in {shards_dir}")


    config = miner.Miner.miner_config()
    comms_ = comms.Comms(  
        wallet=None,
        config=config,
        neuid=config.netuid,
        hparams=hparams.load_hparams(), # use_local_run_hparams=self.config.local)
    )
    bucket = comms_.get_own_bucket("dataset", "read")
    tokens_file, ids_file = sharded_dataset.SharedShardedDataset.locate_shards(0)
    await asyncio.create_task(comms_.s3_get_object(
        tokens_file,
        bucket,
        load_data=False,
    ))        

    args.r2_endpoint_url = f"https://{args.r2_endpoint_url}.r2.cloudflarestorage.com"
    # print("R2 mode enabled. Will upload to R2 bucket.")
    session = boto3.session()
    s3_client = session.client(
        "s3",
        endpoint_url=args.r2_endpoint_url,
        aws_access_key_id=args.r2_access_key_id,
        aws_secret_access_key=args.r2_secret_access_key,
        region_name="auto",
    )

    # Check existing shards in R2
    try:
        response = s3_client.list_objects_v2(
            Bucket=args.r2_bucket, Prefix=os.path.join(args.r2_prefix, "train_")
        )
        shards = len(response.get("Contents", []))
        print(f"Shards found: {shards}")
    except Exception as e:
        print(f"Could not list objects in R2 bucket: {e}")
        existing_count = 0

    

    # ── 1. Size calculation ─────────────────────────────────────────────
    # print("Calculating total size of the dataset …")
    # total_tokens = sum(np.load(f, mmap_mode="r").shape[0] for f in files)
    # total_samples = total_tokens // seq_len
    # print(f"  » tokens  : {total_tokens:,}")
    # print(f"  » samples : {total_samples:,}")

    # # ── 2. Concatenate shards into tokens.bin ───────────────────────────
    # print(f"\n[1/2] Writing {tokens_file} …")
    # t0 = time.perf_counter()
    # tokens_mmap = np.memmap(
    #     tokens_file, dtype=token_dtype, mode="w+", shape=(total_tokens,)
    # )




    # cursor = 0
    file_getter = asyncio.create_task(time.sleep(0))
    for i in range(shards): #(files, start=1):
        await file_getter
        #     shard = np.load(f_path)
        #     n = shard.shape[0]
        #     print(f"  • shard {i:>3}/{len(files)}  ({n:,} tokens)  → offset {cursor:,}")
        #     tokens_mmap[cursor : cursor + n] = shard
        #     cursor += n
        #     del shard
        #     gc.collect()

        # tokens_mmap.flush()
        # del tokens_mmap
        # print(f"tokens.bin written in {time.perf_counter() - t0:.1f}s")

        # ── 3. Compute sample IDs ───────────────────────────────────────────
        # print(f"\n[2/2] Generating {ids_file} …")
        t1 = time.perf_counter()
        
        if i+1 < shards:
            tokens_file, ids_file = sharded_dataset.SharedShardedDataset.locate_shards(i + 1)
            file_getter = asyncio.create_task(comms_.s3_get_object(
                tokens_file,
                bucket,
                load_data=False,
            ))    

        


        # Output paths
        # tokens_file = shards_dir / f"train_{i:06d}.npy"
        # ids_file = shards_dir / f"sample_ids_{i:06d}.bin"

        import pdb
        # pdb.set_trace()

        tokens_view = np.memmap(tokens_file, dtype=token_dtype, mode="r")
        tok_u32 = tokens_view.view(np.uint32)  # reinterpret for 4-byte hashing
        total_samples = 10 * 1024**3
        # sample_ids = np.empty(total_samples, dtype=np.uint64)

        # arr = next(iter(c.partition_all(seq_len, tok_u32)))
        # print(arr)
        # _ = cc.pipe(
        #     tqdm(
        #         c.partition_all(seq_len, tok_u32),
        #         total=tok_u32.shape[0] // seq_len
        #     ),
        #     # lambda x: next(iter(x)),
        #     # [arr],
        #     # cc.partition_all(seq_len),
        #     # cc.map(passthrough_print),
        #     # cc.map(lambda x: list(np.stack(x))),
        #     # cc.map(passthrough_print),
        #     p_map(
        #         cc.compose_left(
        #             # cc.first,
        #             np.stack,
        #             # passthrough_print,
        #             # list,
        #             tokens_handler,
        #             # list,
        #             # passthrough_print,
        #         ),
        #     ),
        #     # cc.concat,
        #     # cc.map(passthrough_print),
        #     list,
        #     cc.map(c.curry(np.stack, dtype=np.uint64)),
        #     cc.map(lambda x: x.tofile(ids_file)),
        #     list,
        # )

        bits = Parallel(n_jobs=os.cpu_count() * 2, prefer="threads")(
            delayed(tokens_handler)(np.stack(arr))
            for arr in tqdm(
                c.partition_all(seq_len, tok_u32), total=tok_u32.shape[0] // seq_len
            )
        )
        sample_ids = np.stack(bits).view(np.uint64)

        buffer = io.BytesIO()
        np.save(buffer, sample_ids)
        buffer.seek(0)

        key = os.path.join(args.r2_prefix, ids_file)
        s3_client.upload_fileobj(buffer, args.r2_bucket, key)
        tqdm.write(
            f"Uploaded sample_ids {i} to R2: s3://{args.r2_bucket}/{key} "
        )


        sample_ids.tofile(ids_file)

        # for i in range(total_samples):
        #     start = i * seq_len
        #     end = start + seq_len
        #     h = hashlib.blake2b(tok_u32[start:end].tobytes(), digest_size=8)
        #     unpack = struct.unpack("<Q", h.digest())[0]
        #     sample_ids[i] = unpack
        #     print(unpack)

        #     # Simple progress ticker every 1 % (optional)
        #     if (i + 1) % max(1, total_samples // 100) == 0:
        #         pct = (i + 1) / total_samples * 100
        #         print(f"\r  • {pct:6.2f}% ", end="", flush=True)

        # print()  # newline after progress bar
        # sample_ids.tofile(ids_file)
        del tokens_view, sample_ids
    print(f"sample_ids.bin written in {time.perf_counter() - t1:.1f}s")

    # ── 4. Integrity summary and validation ─────────────────────────────
    print("\n🔍  Verifying output files …")

    def sha256_file(path: Path, chunk_bytes: int = 64 << 20) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_bytes):
                h.update(chunk)
        return h.hexdigest()

    # Calculate checksums and counts
    t_sha = sha256_file(tokens_file)
    t_count = os.path.getsize(tokens_file) // np.dtype(token_dtype).itemsize
    i_sha = sha256_file(ids_file)
    i_count = os.path.getsize(ids_file) // np.dtype(np.uint64).itemsize

    print(f"tokens.bin SHA-256     : {t_sha}  ({t_count:,} elems)")
    print(f"sample_ids.bin SHA-256 : {i_sha}  ({i_count:,} elems)")

    # Validation checks
    checks = [
        (t_sha, EXPECTED["tokens"]["sha256"], "tokens.bin sha256"),
        (t_count, EXPECTED["tokens"]["elements"], "tokens.bin count"),
        (i_sha, EXPECTED["sample_ids"]["sha256"], "sample_ids.bin sha256"),
        (i_count, EXPECTED["sample_ids"]["elements"], "sample_ids.bin count"),
    ]

    all_ok = True
    for actual, expect, label in checks:
        ok = actual == expect
        all_ok &= ok
        print(f"{label:25}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  expected: {expect}")
            print(f"  actual  : {actual}")

    if all_ok:
        print("\n✅  Preprocessing complete!")
    else:
        print("\n❌  Preprocessing failed validation!")

    return all_ok


async def main():
    parser = argparse.ArgumentParser(
        description="Consolidate .npy shards into tokens.bin and sample_ids.bin",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/shadeform/datasets/smalldclm",
        help="Directory containing train_*.npy shards",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Model context length for sequence chunking",
    )

    parser.add_argument(
        "--token_dtype",
        type=str,
        default="uint16",
        choices=["uint8", "uint16", "uint32"],
        help="Data type used in the shards",
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

    args = parser.parse_args()

    # Convert string dtype to numpy dtype
    token_dtype = getattr(np, args.token_dtype)

    # Validate inputs
    # if not Path(args.data_root).exists():
    #     raise FileNotFoundError(f"Shards directory does not exist: {args.data_root}")

    # if args.seq_len <= 0:
    #     raise ValueError("Sequence length must be positive")

    print("Configuration:")
    print(f"  • Shards path: {args.data_root}")
    print(f"  • Sequence length: {args.seq_len}")
    print(f"  • Token dtype: {args.token_dtype}")
    print()

    success = await run_preprocessing(args)#args.data_root, args.seq_len, token_dtype)

    if not success:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
