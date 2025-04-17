# ruff: noqa
"""
Compare performance & output‑equivalence of the two R2 dataset loaders.

Usage
-----
$ python scripts/benchmarks/benchmark_loader_compare.py
"""

import os, asyncio, psutil, torch, json
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd, numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt, seaborn as sns

# --- env --------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]  # project root
load_dotenv(REPO / ".env", override=True)

# sanity‑check creds once so we fail early
for k in [
    "R2_DATASET_ACCOUNT_ID",
    "R2_DATASET_BUCKET_NAME",
    "R2_DATASET_READ_ACCESS_KEY_ID",
    "R2_DATASET_READ_SECRET_ACCESS_KEY",
]:
    assert os.getenv(k), f"env var {k} missing"

# --- tplr -------------------------------------------------------------------
import tplr
from tplr.logging import logger, debug, T
from tplr.r1_old import R2DatasetLoader as R1Loader
from tplr.r2_dataset import R2DatasetLoader as R2Loader

debug()  # verbose logs if you want them – comment‑out for silence


# --------------------------------------------------------------------------- #
N_COMPARE_BATCHES = 3  # batches per iteration whose tokens are compared
ITERATIONS = 3
CONFIGS = [  # (n_pages, batch, seq_len) – edit at will
    (1, 4, 512),
    (1, 8, 512),
    (4, 4, 512),
    (5, 6, 2048),
]

tokenizer = tplr.load_hparams().tokenizer
PROC = psutil.Process()

OUT_DIR = REPO / "scripts" / "benchmarks" / "compare_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "loader_compare_results.csv"


# --------------------------------------------------------------------------- #
async def run_one_loader(
    LoaderCls,
    pages,
    batch_size,
    seq_len,
    tag,
    iteration,
):
    """Create loader, iterate N_COMPARE_BATCHES batches, collect metrics."""
    rss_before = PROC.memory_info().rss / 1_048_576  # MiB
    t0 = T()
    loader = await LoaderCls.create(
        batch_size=batch_size,
        sequence_length=seq_len,
        pages_info=pages,
        tokenizer=tokenizer,
        pack_samples=True,
    )
    create_time = T() - t0

    toks = 0
    batches = []
    t1 = T()
    for i, batch in enumerate(loader):
        if isinstance(batch, np.ndarray):
            batch = torch.tensor(batch)
        toks += batch.numel()
        if i < N_COMPARE_BATCHES:
            batches.append(batch.clone())  # keep for later diff'ing
        if i + 1 >= N_COMPARE_BATCHES:
            break
    proc_time = T() - t1
    rss_after = PROC.memory_info().rss / 1_048_576
    mem_used = rss_after - rss_before

    return dict(
        tag=tag,
        iteration=iteration,
        create_s=create_time,
        iterate_s=proc_time,
        tokens=toks,
        tps=toks / proc_time if proc_time else 0,
        mem_mb=mem_used,
        batches=batches,  # for equality test
    )


async def benchmark():
    metrics = []

    for n_pages, batch_size, seq_len in CONFIGS:
        for it in range(ITERATIONS):
            # deterministic page list – assert both helpers agree
            seed = it  # seed must be int
            pages_r1 = await R1Loader.next_pages(offset=it, n_pages=n_pages, seed=seed)
            pages_r2 = await R2Loader.next_pages(offset=it, n_pages=n_pages, seed=seed)
            assert pages_r1 == pages_r2, (
                "next_pages mismatch, seed logic diverged:\n"
                f"r1: {pages_r1[:3]} …\n"
                f"r2: {pages_r2[:3]} …"
            )
            pages = pages_r1  # use for both loaders

            r1 = await run_one_loader(
                R1Loader, pages, batch_size, seq_len, "r1_old", it
            )
            r2 = await run_one_loader(
                R2Loader, pages, batch_size, seq_len, "r2_new", it
            )

            # --- correctness check ---------------------------------------
            for b1, b2 in zip(r1["batches"], r2["batches"]):
                if not torch.equal(b1, b2):
                    # dump a repro snippet + fail hard
                    repro = dict(pages=pages, batch_size=batch_size, seq_len=seq_len)
                    (OUT_DIR / "mismatch_repro.json").write_text(
                        json.dumps(repro, indent=2)
                    )
                    raise ValueError(
                        "Token mismatch between loaders "
                        f"(config={n_pages, batch_size, seq_len}, iter={it})"
                    )

            # strip large tensors before log / csv
            r1.pop("batches"), r2.pop("batches")
            r1.update(dict(n_pages=n_pages, batch_size=batch_size, seq_len=seq_len))
            r2.update(dict(n_pages=n_pages, batch_size=batch_size, seq_len=seq_len))
            metrics.extend([r1, r2])

            logger.info(
                f"[cmp] pages={n_pages} batch={batch_size} seq={seq_len} "
                f"iter={it}: r1 {r1['tps']:.0f} tok/s | r2 {r2['tps']:.0f} tok/s"
            )

    return pd.DataFrame(metrics)


def plot(df: pd.DataFrame):
    """Tiny helper – writes a couple of comparison plots."""
    for metric, title in [
        ("tps", "Tokens/s"),
        ("create_s", "Loader‑create time (s)"),
        ("iterate_s", "Iterate time (s)"),
        ("mem_mb", "ΔRSS (MiB)"),
    ]:
        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=df,
            x="tag",
            y=metric,
            hue="n_pages",
            errorbar="sd",  # new API
        )
        plt.title(title)
        plt.savefig(OUT_DIR / f"{metric}.png")
        plt.close()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    df = asyncio.run(benchmark())
    df.to_csv(CSV_PATH, index=False)
    print("\n== raw metrics saved to", CSV_PATH)
    print(df.groupby(["tag"]).agg(dict(tps=["mean", "std"], mem_mb="mean")).round(2))

    try:
        plot(df)
        print(f"plots written → {OUT_DIR}")
    except Exception as e:
        print("plotting failed:", e)
