# DCLM Optimized Dataset Setup

This guide explains how to properly download, validate, and set up the optimized DCLM dataset for use with your miner. The dataset will be downloaded directly from Hugging Face and transferred to your Cloudflare R2 bucket with improved optimizations for performance and reliability.

## Dataset Information

The DCLM dataset is based on the [mlfoundations/dclm-baseline-1.0-parquet](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet) dataset on Hugging Face, part of the research available at [DataComp](https://data.commoncrawl.org/contrib/datacomp/index.html).

Our optimized version includes:

- ZSTD compression with optimal settings
- Improved row group sizes for better query performance
- Optimized metadata for faster access
- Verified file integrity at both header and footer levels

## System Requirements

The dataset transfer requires adequate bandwidth. Files are temporarily processed on your machine but not permanently stored locally. Higher internet speeds and more worker threads will significantly accelerate the process.

### Recommended Hardware

- **Network**: 1gbps+
- **Local Storage**: 100GB (for temporary processing)
- **RAM**: 4GB+
- **CPU Cores**: 8+ (more cores = faster processing)
- **Estimated Download Time**: 15-24 hours (depending on network speed)

## Setup Instructions

### 1. Set Up Your Cloudflare R2 Bucket

1. Create a bucket named `dataset` in your Cloudflare R2 account
2. Create appropriate API keys:
   - A set of keys with write access (for uploading the dataset)
   - A set of read-only keys (for your miner to access the data)

### 2. Download and Configure the HFDownloader Utility

Our modern downloader utility offers high-performance, multi-threaded transfers with automatic retries and resumption capabilities.

```bash
# Clone the templar repository if you don't already have it
git clone git@github.com:tplr-ai/templar.git
cd templar/scripts
chmod +x hfdownloader.py run_hfdownloader.sh
```

### 3. Start the Dataset Transfer with Enhanced Resilience

The `run_hfdownloader.sh` wrapper script provides automatic retries with exponential backoff if any errors occur during the download process.

```bash
# Configure dataset bucket name to use
export DATABUCKET="dataset"

./run_hfdownloader.sh \
  --dataset="mlfoundations/dclm-baseline-1.0-parquet" \
  --branch="main" \
  --r2-account="YOUR_ACCOUNT_ID" \
  --r2-key="YOUR_WRITE_ACCESS_KEY" \
  --r2-secret="YOUR_WRITE_SECRET_KEY" \
  --r2-bucket="$DATABUCKET" \
  --workers=16 \
  --r2-subfolder="mlfoundations-dclm-baseline-1.0-parquet-optimized" \
  --compression-level=3 \
  --row-group-size=1000
```

Key parameters explained:

- `--workers`: Number of concurrent download threads (adjust based on your CPU cores)
- `--compression-level`: ZSTD compression level (1-22, higher = better compression but slower)
- `--row-group-size`: Parquet row group size optimization (impacts query performance)

The download process will:

1. Build a cache of existing files in your R2 bucket
2. Download files from Hugging Face in parallel
3. Optimize parquet files with ZSTD compression and row group sizing
4. Upload optimized files to your R2 bucket
5. Verify file integrity after upload
6. Track progress to allow resuming if interrupted

### 4. Validate Dataset Integrity

For dataset validation, simply run the downloader again. It will automatically:

- Check for missing files
- Verify the integrity of existing files
- Re-download any corrupted files
- Complete the dataset if any files are missing

```bash
./run_hfdownloader.sh \
  --dataset="mlfoundations/dclm-baseline-1.0-parquet" \
  --branch="main" \
  --r2-account="YOUR_ACCOUNT_ID" \
  --r2-key="YOUR_WRITE_ACCESS_KEY" \
  --r2-secret="YOUR_WRITE_SECRET_KEY" \
  --r2-bucket="$DATABUCKET" \
  --workers=16 \
  --r2-subfolder="mlfoundations-dclm-baseline-1.0-parquet-optimized" \
  --hf-prefix="filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data" \
  --compression-level=3 \
  --row-group-size=1000
```

For extensive validation, you can also use the `--check-corrupted` flag:

```bash
./scripts/hfdownloader.py \
  --r2-account="YOUR_ACCOUNT_ID" \
  --r2-key="YOUR_WRITE_ACCESS_KEY" \
  --r2-secret="YOUR_WRITE_SECRET_KEY" \
  --r2-bucket="$DATABUCKET" \
  --r2-subfolder="mlfoundations-dclm-baseline-1.0-parquet-optimized" \
  --hf-prefix="filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data" \
  --check-corrupted \
  --workers=16
```

### 5. Update Metadata Files for Your Bucket

After the dataset transfer is complete, you must update and upload the metadata files to your R2 bucket. These files are crucial for the miner to locate and access the dataset.

The `_shard_sizes.json` file contains paths that need to be updated to reference your specific bucket name:

```bash
# Return to the templar repo root
cd ~/templar

# Update the _shard_sizes.json to use your bucket name
sed -i 's|"dataset/|"'$DATABUCKET'/|g' _shard_sizes.json
```

### 6. Upload Metadata Files to Your R2 Bucket

After updating the metadata files, you need to upload them to your R2 bucket in the correct location:

```python
import boto3
import os

# Configure R2 client
s3 = boto3.client(
    service_name="s3",
    endpoint_url=f"https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com",
    aws_access_key_id="YOUR_WRITE_ACCESS_KEY",
    aws_secret_access_key="YOUR_WRITE_SECRET_KEY",
    region_name="auto"
)

# Upload metadata files
with open("_shard_sizes.json", "rb") as f:
    s3.upload_fileobj(f, os.environ.get("DATABUCKET", "dataset"), 
                      "mlfoundations-dclm-baseline-1.0-parquet-optimized/_shard_sizes.json")

with open("_metadata.yaml", "rb") as f:
    s3.upload_fileobj(f, os.environ.get("DATABUCKET", "dataset"), 
                      "mlfoundations-dclm-baseline-1.0-parquet-optimized/_metadata.yaml")
```

You can also manually upload these files through the Cloudflare R2 dashboard.

### 7. Clear Local Cache (Required)

After uploading the metadata files, you must clear your local cache to force the miner to download the new metadata on its next run:

```bash
rm -rf ./.cache/tplr/*
```

### 8. Configure Your Miner to Use the Dataset

Set these environment variables for your miner to connect to your Cloudflare R2 bucket:

```bash
export R2_DATASET_ACCOUNT_ID=YOUR_ACCOUNT_ID
export R2_DATASET_BUCKET_NAME=$DATABUCKET
export R2_DATASET_READ_ACCESS_KEY_ID=YOUR_READ_ACCESS_KEY
export R2_DATASET_READ_SECRET_ACCESS_KEY=YOUR_READ_SECRET_KEY
```

**Note**: For security, create separate read-only API keys for your miner. Never use your write access keys for the miner.

## Troubleshooting

### Common Issues

1. **Network Interruptions**: The wrapper script will automatically retry with exponential backoff. It will make up to 30 attempts with a maximum wait time of 2 minutes between retries.

2. **Corrupted Files**: If you encounter corrupted files, run the validation command with the `--check-corrupted` flag. This will scan all files and automatically remove and re-download any corrupted ones.

3. **Slow Downloads**: Consider:
   - Increasing `--workers` if you have more CPU cores available
   - Adjusting compression level (lower = faster uploads but larger files)
   - Using a machine with higher network bandwidth

4. **Space Issues**: If you run out of space during temporary file processing, ensure you have at least 100GB of free space or use a machine with more storage.

5. **HTTP Timeout Errors**: These are handled automatically with retries. If you see persistent timeouts, there might be network connectivity issues to either Hugging Face or Cloudflare.

6. **Metadata File Issues**: If your miner can't find the dataset:
   - Ensure that `_shard_sizes.json` and `_metadata.yaml` were correctly uploaded to the bucket
   - Verify that the paths in these files correctly reference your bucket name
   - Check that the paths point to the optimized dataset folder path
   - Clear your local cache after uploading the updated files

## Performance Considerations

- **Optimizing Download Speed**: The parallel download architecture can efficiently utilize up to 64 concurrent connections
- **Memory Usage**: Each worker requires approximately 256MB of memory buffer
- **Storage I/O**: Using SSD storage for temporary files will significantly improve performance
- **Network Utilization**: The downloader is designed to efficiently utilize available bandwidth with adaptive throttling

This optimized dataset provides significant performance improvements when used with your miner, ensuring faster loading times and more efficient processing of training data.
