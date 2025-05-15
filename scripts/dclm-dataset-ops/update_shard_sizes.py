#!/usr/bin/env python3
"""
update_shard_sizes.py

Load a JSON file describing dataset shards, compute the total number of rows
per shard group, and inject a "total_rows" field into each group.
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON data from the given file path."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_totals(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    For each top-level key in data, sum its shards' 'num_rows'
    and rebuild the dict so 'total_rows' comes right after 'split'.
    """
    updated: Dict[str, Any] = {}
    for group, info in data.items():
        shards = info.get("shards", [])
        total_rows = sum(shard.get("num_rows", 0) for shard in shards)
        # Reorder keys: split, total_rows, shards
        updated[group] = {
            "split": info.get("split"),
            "total_rows": total_rows,
            "shards": shards,
        }
    return updated


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Write JSON data to the given file path with pretty indentation."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Updated totals written to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute and inject 'total_rows' into a shard-sizes JSON."
    )
    parser.add_argument("input", type=Path, help="Path to your _shard_sizes.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to write updated JSON (defaults to overwrite input)",
    )
    args = parser.parse_args()

    data = load_json(args.input)
    updated = compute_totals(data)
    out_path = args.output or args.input
    save_json(updated, out_path)


if __name__ == "__main__":
    main()
