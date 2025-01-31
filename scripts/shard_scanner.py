import os
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory (templar folder)
PROJECT_ROOT = Path(__file__).parent.parent
ENV_PATH = PROJECT_ROOT / '.env'

# Load environment variables from .env file
load_dotenv(ENV_PATH)

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
    Stores relative paths instead of full bucket paths for better portability.
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
    #    Store (cfg_name, split, file_path, relative_path) for grouping results
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
                # Extract relative path by removing bucket prefix
                relative_path = fp.replace(f"{r2_bucket}/", "", 1)
                all_tasks.append((cfg_name, split, fp, relative_path))

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
                cfg_name, split, fp, relative_path = futures_map[future]
                pbar.update(1)
                try:
                    rowcount = future.result()
                    shard_sizes[cfg_name]["shards"].append(
                        {"path": relative_path, "num_rows": rowcount}
                    )
                    shard_sizes[cfg_name]["total_rows"] += rowcount
                    shard_sizes[cfg_name]["split"] = split
                except Exception as e:
                    shard_sizes[cfg_name]["shards"].append(
                        {"path": relative_path, "num_rows": 0, "error": str(e)}
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
    # Construct R2 endpoint using account ID
    account_id = os.getenv('R2_DATASET_ACCOUNT_ID')
    r2_endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    
    R2_CREDS = {
        "r2_bucket": os.getenv('R2_DATASET_BUCKET_NAME'),
        "r2_endpoint": r2_endpoint,
        "r2_access_key": os.getenv('R2_DATASET_READ_ACCESS_KEY_ID'),
        "r2_secret_key": os.getenv('R2_DATASET_READ_SECRET_ACCESS_KEY'),
        "local_metadata_path": str(PROJECT_ROOT / "metadata.yaml"),
        "local_output_path": str(PROJECT_ROOT / "shard_sizes.json"),
    }

    # Validate environment variables
    missing_vars = [k for k, v in R2_CREDS.items() if v is None]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")

    print("Starting shard counting with configuration:")
    print(f"- Bucket: {R2_CREDS['r2_bucket']}")
    print(f"- Endpoint: {R2_CREDS['r2_endpoint']}")
    print(f"- Metadata path: {R2_CREDS['local_metadata_path']}")
    print(f"- Output path: {R2_CREDS['local_output_path']}")

    await precompute_shard_counts(**R2_CREDS)


if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)