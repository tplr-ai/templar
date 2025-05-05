#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [ ]
# ///

"""
Convert a CSV of Parquet shards into our standard _shard_sizes.json format.

Assumes your CSV has columns:
  - filename
  - filepath
  - row_count
  - byte_size
  - file_hash

Adjust `INPUT_CSV` and `OUTPUT_JSON` as needed.
"""

import csv
import json
import os
import sys
from collections import defaultdict

# Path to your input CSV
INPUT_CSV = "dclm-dataset.csv"

# Path to write the JSON manifest
OUTPUT_JSON = "_shard_sizes.json"

# Fixed dataset root to prefix every shard path
DATASET_ROOT = "dataset/dclm-dataset"

def build_manifest(csv_path: str) -> dict:
    """
    Read the CSV and produce a dict like:
    {
      "default": { "split": None, "shards": [], "total_rows": 0 },
      "global-shard_01_of_10": {
         "split": None,
         "shards": [ {...}, {...}, ... ]
      },
      ...
    }
    """
    groups = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = row["filepath"]
            global_shard = filepath.split("/")[0]

            full_path = os.path.join(DATASET_ROOT, filepath)

            groups[global_shard].append(
                {
                    "path": full_path,
                    "num_rows": int(row["row_count"]),
                    "byte_size": int(row["byte_size"]),
                    "file_hash": row["file_hash"],
                }
            )

    manifest = {"default": {"split": None, "shards": [], "total_rows": 0}}

    for shard_name, shard_list in sorted(groups.items()):
        manifest[shard_name] = {
            "split": None,
            "shards": shard_list,
        }

    return manifest


def main():
    if not os.path.isfile(INPUT_CSV):
        print(f"ERROR: Input CSV not found at '{INPUT_CSV}'", file=sys.stderr)
        sys.exit(1)

    manifest = build_manifest(INPUT_CSV)

    # write to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
        json.dump(manifest, out, indent=2, ensure_ascii=False)

    print(f"Written JSON manifest with {len(manifest)-1} shards to '{OUTPUT_JSON}'")


if __name__ == "__main__":
    main()
