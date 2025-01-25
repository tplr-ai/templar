"""
shard_scanner.py - Parallel Parquet Row Counter

Features:
- Reads local YAML metadata
- Processes R2 Parquet files with read-only access
- Uses parallel threading for faster processing
- Detailed progress tracking with tqdm
- Writes results to local JSON
"""

import asyncio
import json
import s3fs
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

async def precompute_shard_counts(
    r2_bucket: str,
    r2_endpoint: str,
    r2_access_key: str,
    r2_secret_key: str,
    local_metadata_path: str,
    local_output_path: str,
):
    # Initialize S3 connection
    fs = s3fs.S3FileSystem(
        key=r2_access_key,
        secret=r2_secret_key,
        client_kwargs={"endpoint_url": r2_endpoint},
    )

    # Load metadata with progress
    def _load_metadata():
        with open(local_metadata_path, "r") as f:
            return yaml.safe_load(f)
    
    import yaml
    metadata = await asyncio.to_thread(_load_metadata)

    # Process configurations with parallel file discovery
    config_dict = {}
    configs = metadata.get("configs", [])
    
    with tqdm(configs, desc="Scanning configs", unit="config") as config_pbar:
        for c in config_pbar:
            cfg_name = c["config_name"]
            config_pbar.set_postfix({"config": cfg_name})
            
            data_files = c.get("data_files", [])
            for df in data_files:
                path_pattern = f"{r2_bucket}/{df['path']}"
                try:
                    files = fs.glob(path_pattern)
                    if not files:
                        tqdm.write(f"\n⚠️  No files found: {path_pattern}")
                        
                    config_dict[cfg_name] = {
                        "split": df["split"],
                        "files": sorted(files),
                    }
                except Exception as e:
                    tqdm.write(f"\n❌ Error scanning {cfg_name}: {str(e)}")

    # Process files with parallel row counting
    shard_sizes = {}
    MAX_WORKERS = 8  # Adjust based on your network capacity
    
    with tqdm(config_dict.items(), desc="Processing configs", unit="config") as main_pbar:
        for cfg_name, info in main_pbar:
            main_pbar.set_postfix({"config": cfg_name})
            split = info["split"]
            files = info["files"]
            
            total_rows = 0
            shards_list = []
            
            # Process files in parallel batches
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                for fp in files:
                    futures.append(executor.submit(
                        lambda f: (f, len(pq.read_table(f, filesystem=fs, columns=["text"]))),
                        fp
                    ))
                
                # Collect results with progress
                with tqdm(futures, desc=f"Files in {cfg_name}", unit="file", leave=False) as file_pbar:
                    for future in file_pbar:
                        try:
                            fp, rowcount = future.result()
                            shards_list.append({"path": fp, "num_rows": rowcount})
                            total_rows += rowcount
                        except Exception as e:
                            tqdm.write(f"\n❌ Error counting {fp}: {str(e)}")
                            shards_list.append({"path": fp, "num_rows": 0, "error": str(e)})

            shard_sizes[cfg_name] = {
                "split": split,
                "shards": shards_list,
                "total_rows": total_rows,
            }

    # Write output
    def _save_output():
        output_path = Path(local_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(shard_sizes, f, indent=2)
    
    await asyncio.to_thread(_save_output)
    print(f"\n✅ Results saved to: {local_output_path}")

async def main():
    # Configuration - fill in your details
    R2_CREDS = {
        "r2_bucket": "80f15715bb0b882c9e967c13e677ed7d",
        "r2_endpoint": "https://80f15715bb0b882c9e967c13e677ed7d.r2.cloudflarestorage.com",
        "r2_access_key": "de1b777bd4e13cd61bb8aeb6ae431865",
        "r2_secret_key": "947906d741cc12eeab5c0c225161c4833bbc932bb3ed61038847f645bbff6eb3",
        "local_metadata_path": "./metadata_updated.yaml",
        "local_output_path": "./new_r2_shard_sizes.json"
    }
    
    await precompute_shard_counts(**R2_CREDS)

if __name__ == "__main__":
    asyncio.run(main())