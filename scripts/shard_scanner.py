async def precompute_shard_counts(
    r2_bucket: str,
    r2_endpoint: str,
    r2_access_key: str,
    r2_secret_key: str,
    local_metadata_path: str,
    local_output_path: str,
):
    """
    Parallel row counting across all configs and files simultaneously,
    using Parquet metadata (no full column read).
    """

    import asyncio
    import json
    import s3fs
    import pyarrow.parquet as pq
    from pathlib import Path
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import yaml

    # 1. Initialize S3 connection
    fs = s3fs.S3FileSystem(
        key=r2_access_key,
        secret=r2_secret_key,
        client_kwargs={"endpoint_url": r2_endpoint},
    )

    # 2. Load metadata
    def _load_metadata():
        with open(local_metadata_path, "r") as f:
            return yaml.safe_load(f)

    metadata = await asyncio.to_thread(_load_metadata)
    configs = metadata.get("configs", [])

    # 3. Create a single global list of tasks across all configs
    #    We'll store (cfg_name, split, file_path) so we can group results later
    all_tasks = []

    for c in configs:
        cfg_name = c["config_name"]
        split = None
        data_files = c.get("data_files", [])
        for df in data_files:
            path_pattern = f"{r2_bucket}/{df['path']}"
            files = fs.glob(path_pattern)
            split = df["split"]
            for fp in sorted(files):
                all_tasks.append((cfg_name, split, fp))

    # 4. Prepare data structures for storing results
    shard_sizes = {}
    for c in configs:
        cfg_name = c["config_name"]
        shard_sizes[cfg_name] = {
            "split": None,
            "shards": [],
            "total_rows": 0,
        }

    # 5. Define a function to count rows via file metadata
    def count_rows_via_metadata(fp):
        parquet_file = pq.ParquetFile(fp, filesystem=fs)
        return parquet_file.metadata.num_rows

    # 6. Spawn parallel tasks for the entire list
    MAX_WORKERS = 64  # Increase if your system can handle it
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures_map = {
            executor.submit(count_rows_via_metadata, task[2]): task
            for task in all_tasks
        }

        with tqdm(total=len(futures_map), desc="All files", unit="file") as pbar:
            for future in as_completed(futures_map):
                cfg_name, split, fp = futures_map[future]
                pbar.update(1)
                try:
                    rowcount = future.result()
                    shard_sizes[cfg_name]["shards"].append(
                        {"path": fp, "num_rows": rowcount}
                    )
                    shard_sizes[cfg_name]["total_rows"] += rowcount
                    shard_sizes[cfg_name]["split"] = split
                except Exception as e:
                    shard_sizes[cfg_name]["shards"].append(
                        {"path": fp, "num_rows": 0, "error": str(e)}
                    )

    # 7. Save results to JSON
    def _save_output():
        output_path = Path(local_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(shard_sizes, f, indent=2)

    await asyncio.to_thread(_save_output)
    print(f"\n✅ Results saved to: {local_output_path}")


async def main():
    R2_CREDS = {
        "r2_bucket": "",
        "r2_endpoint": "https://.r2.cloudflarestorage.com",
        "r2_access_key": "",
        "r2_secret_key": "",
        "local_metadata_path": "./metadata.yaml",
        "local_output_path": "./shard_sizes.json",
    }
    await precompute_shard_counts(**R2_CREDS)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
