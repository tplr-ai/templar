# ruff : noqa

"""
Benchmark the original R2DatasetLoader (tplr.r2_dataset)
vs. the patched version living in tplr.d2.

Metrics captured per run:
    • pages_fetch_time
    • loader_creation_time
    • processing_time       (iterating through N batches)
    • tokens_per_second
    • memory_used_mb
    • total_duration

Environment:
    – Requires the same .env the rest of tplr expects (R2 creds, etc.)
    – Expects both modules to import cleanly.

TODO:
    • Add GPU utilisation if torch.cuda.is_available().
    • Make page/batch/seq param grid configurable via JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Type

import numpy as np
import pandas as pd
import psutil
import torch
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
#  Load environment (.env in project root)                                    #
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]  # project root (adjust if needed)
load_dotenv(ROOT / ".env", override=True)

# --------------------------------------------------------------------------- #
#  Import the two loaders                                                     #
# --------------------------------------------------------------------------- #
try:
    from tplr.r2_dataset_old import R2DatasetLoader as _OldLoader  # original
except Exception as e:
    raise ImportError(f"Could not import tplr.r2_dataset_old.R2DatasetLoader: {e}")

# Allow either `R2DatasetLoader` *or* `D2DatasetLoader` in tplr.d2
try:
    from tplr.r2_dataset import R2DatasetLoader as _NewLoader  # patched
except ImportError:
    from tplr.r2_dataset import D2DatasetLoader as _NewLoader

# --------------------------------------------------------------------------- #
#  Shared helpers                                                             #
# --------------------------------------------------------------------------- #
import tplr

HPARAMS = tplr.load_hparams()
TOKENIZER = HPARAMS.tokenizer


def now() -> float:
    return time.perf_counter()


async def run_one_iteration(
    loader_cls: Type,
    *,
    offset: int,
    n_pages: int,
    batch_size: int,
    sequence_length: int,
    max_batches: int,
) -> Dict:
    """One full loop: pages → loader → iterate ≤ max_batches batches."""
    # Fetch pages
    t0 = now()
    pages = await loader_cls.next_pages(
        offset=offset, n_pages=n_pages, seed="benchmark_d2_vs_r2"
    )
    pages_fetch_time = now() - t0

    # Create loader
    t0 = now()
    loader = await loader_cls.create(
        batch_size=batch_size,
        sequence_length=sequence_length,
        pages_info=pages,
        tokenizer=TOKENIZER,
        pack_samples=True,
    )
    loader_creation_time = now() - t0

    # Iterate
    proc_start = now()
    total_tokens = 0
    n_batches = 0
    for batch in loader:
        # bail early if we just want “warm” batches
        if n_batches >= max_batches:
            break
        if isinstance(batch, np.ndarray):
            total_tokens += batch.size
        elif torch.is_tensor(batch):
            total_tokens += batch.numel()
        else:
            # fallback – assume flat list of ints
            total_tokens += len(batch)
        n_batches += 1
    processing_time = now() - proc_start

    tokens_per_second = (
        total_tokens / processing_time if processing_time > 0 else float("nan")
    )

    return {
        "pages": n_pages,
        "batch_size": batch_size,
        "seq_len": sequence_length,
        "total_tokens": total_tokens,
        "num_batches": n_batches,
        "pages_fetch_time": pages_fetch_time,
        "loader_creation_time": loader_creation_time,
        "processing_time": processing_time,
        "tokens_per_second": tokens_per_second,
    }


async def benchmark_loader(
    name: str,
    loader_cls: Type,
    *,
    page_counts: List[int],
    iterations: int,
    batch_size: int,
    sequence_length: int,
    max_batches: int,
) -> pd.DataFrame:
    """Run the grid for a single loader implementation."""
    results: List[Dict] = []
    proc = psutil.Process()

    for n_pages in page_counts:
        for it in range(iterations):
            rss_before = proc.memory_info().rss / (1024 * 1024)  # MB
            t0 = now()
            metrics = await run_one_iteration(
                loader_cls,
                offset=it,
                n_pages=n_pages,
                batch_size=batch_size,
                sequence_length=sequence_length,
                max_batches=max_batches,
            )
            total_duration = now() - t0
            rss_after = proc.memory_info().rss / (1024 * 1024)
            metrics.update(
                {
                    "impl": name,
                    "iteration": it,
                    "total_duration": total_duration,
                    "memory_used_mb": rss_after - rss_before,
                }
            )
            results.append(metrics)
            print(
                f"[{name}] pages={n_pages:2d} iter={it} "
                f"{metrics['tokens_per_second']:.0f} tok/s "
                f"ΔRSS={metrics['memory_used_mb']:.1f} MB"
            )
    return pd.DataFrame(results)


# --------------------------------------------------------------------------- #
#  CLI / main                                                                 #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark tplr.r2_dataset vs tplr.d2 loaders."
    )
    p.add_argument(
        "--iterations", type=int, default=3, help="Iterations per page count"
    )
    p.add_argument(
        "--page-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="List of page counts to test",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument(
        "--max-batches",
        type=int,
        default=8,
        help="Batches to process per iteration (limits runtime)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/loader_benchmark_results.csv"),
        help="CSV output path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    async def _run() -> None:
        df_old = await benchmark_loader(
            "old",
            _OldLoader,
            page_counts=args.page_counts,
            iterations=args.iterations,
            batch_size=args.batch_size,
            sequence_length=args.seq_len,
            max_batches=args.max_batches,
        )
        df_new = await benchmark_loader(
            "new",
            _NewLoader,
            page_counts=args.page_counts,
            iterations=args.iterations,
            batch_size=args.batch_size,
            sequence_length=args.seq_len,
            max_batches=args.max_batches,
        )
        df = pd.concat([df_old, df_new], ignore_index=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nSaved results → {args.out}")

        # quick aggregate
        summary = (
            df.groupby(["impl", "pages"])[
                ["tokens_per_second", "total_duration", "memory_used_mb"]
            ]
            .mean()
            .round(2)
        )
        print("\n=== mean across runs ===")
        print(summary)

    # uvloop is nice-to-have
    try:
        import uvloop

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass

    asyncio.run(_run())


if __name__ == "__main__":
    main()
