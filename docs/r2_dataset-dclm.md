# DCLM Dataset Setup for Miners

This guide explains how to properly download, validate, and set up the DCLM dataset for use with your miner. The miner will download and process the dataset during its operation, transferring data from Hugging Face to your Cloudflare R2 bucket.

## Dataset Information

The DCLM dataset is based on the [mlfoundations/dclm-baseline-1.0-parquet](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet) dataset on Hugging Face. This is part of the research available at [DataComp](https://data.commoncrawl.org/contrib/datacomp/index.html).

## System Requirements

You will need to make sure you have adequate bandwidth. The dataset will be transferred through your machine but NOT be stored locally by default. The higher the internet speed and more workers/cores you can utilize, the faster the dataset will be processed. Consider that this process will take 20-30 hours, so use screen/tmux accordingly.

### Recommended Hardware

- **Network**: 1gbps+
- **Local Storage**: 100GB (for temporary processing)
- **RAM**: 4GB+ (this process is not memory intensive)
- **CPU Cores**: 8+ (more cores = faster processing)
- **Estimated Download Time**: 20-30 hours (depending on network speed)

## Setup Instructions

### 1. Set Up Your Cloudflare R2 Bucket

1. Create a bucket named `dataset` in your Cloudflare R2 account
2. Create appropriate API keys:
   - A set of keys with write access (for uploading the dataset)
   - A set of read-only keys (for your miner to access the data)

### 2. Clone the Correct Downloader Version

> ⚠️ **CRITICAL**: You must use the specific commit version of the downloader. Using the main branch will download the wrong dataset!

```bash
# Clone the repository at the specific commit
git clone https://github.com/distributedstatemachine/HuggingFaceModelDownloader
cd HuggingFaceModelDownloader

# IMPORTANT: Checkout the specific commit for DCLM dataset
git checkout 70b0ce8061ec70af5738fdfbf5ac9e0ae02bdffc
```

### 3. Configure Environment Variables

```bash
# Configure dataset bucket name to use
export DATABUCKET="dataset"

# Gather CPU count to use for transfer
export CPUCOUNT=$(grep -c '^processor' /proc/cpuinfo)

```

### 4. Set Up R2 Credentials

Create a local `.env` file with your R2 write credentials:

```bash
tee .env << 'EOF'
R2_ACCOUNT_ID=your_account_id
R2_WRITE_ACCESS_KEY_ID=your_write_access_key
R2_WRITE_SECRET_ACCESS_KEY=your_write_secret_key
EOF
```

### 5. Install Go (if not already installed)

```bash
# Download and install Go
wget https://go.dev/dl/go1.23.5.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.23.5.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# Make the PATH change permanent
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc

# Verify Go installation
go version
```

### 6. Start the Dataset Transfer

```bash
# Start the download process
go run main.go \
  -d "mlfoundations/dclm-baseline-1.0-parquet" \
  --r2 --skip-local -c $CPUCOUNT \
  --r2-bucket $DATABUCKET

# After the first download completes, run it again to verify and download any missing files
go run main.go \
  -d "mlfoundations/dclm-baseline-1.0-parquet" \
  --r2 --skip-local -c $CPUCOUNT \
  --r2-bucket $DATABUCKET
```

![image](https://github.com/user-attachments/assets/f9235ac7-9861-4253-a7aa-24feca5e96ef)

![image](https://github.com/user-attachments/assets/a99737d1-259e-433d-9dcd-71e921c04e4c)

### 7. Validate the Dataset

**Note: to run the validation script, you need to have [uv installed](https://docs.astral.sh/uv/getting-started/installation/).**

After completing the download, validate that your uploaded dataset matches the expected file sizes and hashes:

```bash
# Clone the templar repository if you don't already have it
git clone git@github.com:tplr-ai/templar.git
cd templar/scripts/dclm-dataset-ops

# Create a local .env file for R2 account credentials
tee .env << 'EOF'
R2_ACCOUNT_ID=your_account_id
R2_READ_ACCESS_KEY_ID=your_read_access_key
R2_READ_SECRET_ACCESS_KEY=your_read_secret_key
EOF

# Run the validation script
./shards_validator.py "../../_shard_sizes.json" \
  --r2-bucket dataset \
  --r2-account-id $R2_ACCOUNT_ID \
  --r2-access-key-id $R2_READ_ACCESS_KEY_ID \
  --r2-access-key-secret $R2_READ_SECRET_ACCESS_KEY
```

![image](https://github.com/user-attachments/assets/0f68675f-d5b7-463b-b06e-318c1b0555c6)

This validates that all shards have been properly uploaded with the correct sizes and hashes. Review the validation results to ensure your dataset is complete and correct.

### 8. Update the _shard_sizes.json `path` values

Before proceeding, you need to modify the `_shard_sizes.json` file to ensure it references your specific bucket name. By default, the paths in this file use `dataset/mlfoundations-dclm-baseline-1.0-parquet/...`, but you need to replace `dataset` with your actual bucket name.

**Option 1: Using sed:**

```bash
# Replace "dataset" with your bucket name
sed -i 's|"dataset/mlfoundations-dclm-baseline-1.0-parquet/|"'$DATABUCKET'/mlfoundations-dclm-baseline-1.0-parquet/|g' _shard_sizes.json
```

**Option 2: Using Python:**

```python
import json

# Load the file
with open("_shard_sizes.json", "r") as f:
    data = json.load(f)

# Replace paths (replace "dataset" with your actual bucket name)
updated_data = {}
for k, v in data.items():
    new_key = k.replace("dataset/", f"{DATABUCKET}/")
    updated_data[new_key] = v

# Save back to file
with open("_shard_sizes.json", "w") as f:
    json.dump(updated_data, f, indent=2)
```

This step is critical to ensure that your miner can properly locate and load the dataset files from your specific bucket.

### 9. Configure Your Miner to Use the Dataset

Set these environment variables for your miner to connect to your Cloudflare R2 bucket:

```bash
export R2_DATASET_ACCOUNT_ID=...
export R2_DATASET_BUCKET_NAME=dataset
export R2_DATASET_READ_ACCESS_KEY_ID=...
export R2_DATASET_READ_SECRET_ACCESS_KEY=...
```

**Note**: For security, create separate read-only API keys for your miner. Never use your write access keys for the miner.

### 10. Upload Metadata Files to Your R2 Bucket

You must upload the `_shard_sizes.json` and `_metadata.yaml` files to your R2 bucket in the `mlfoundations-dclm-baseline-1.0-parquet` directory. You can use one of the following methods:

**Option 1: Using Python boto3:**

```python
import boto3
import json
import os

s3 = boto3.client(
    service_name="s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=os.getenv("R2_WRITE_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_WRITE_SECRET_ACCESS_KEY"),
    region_name="auto"
)

with open("_shard_sizes.json", "rb") as f:
    s3.upload_fileobj(f, DATABUCKET, "mlfoundations-dclm-baseline-1.0-parquet/_shard_sizes.json")

with open("_metadata.yaml", "rb") as f:
    s3.upload_fileobj(f, DATABUCKET, "mlfoundations-dclm-baseline-1.0-parquet/_metadata.yaml")
```

**Option 2: Using rclone:**

```bash
# Configure rclone for your R2 if you haven't done already
# example command to create a config file:
# rclone config create r2 s3 \
#   provider=Cloudflare \
#   access_key_id=$R2_WRITE_ACCESS_KEY_ID \
#   secret_access_key=$R2_WRITE_SECRET_ACCESS_KEY \
#   endpoint=https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com

# Upload files
rclone copy _shard_sizes.json r2-dataset:$DATABUCKET/mlfoundations-dclm-baseline-1.0-parquet/
rclone copy _metadata.yaml r2-dataset:$DATABUCKET/mlfoundations-dclm-baseline-1.0-parquet/
```

**Option 3: Manual Upload via Cloudflare Dashboard:**

1. Log in to your Cloudflare dashboard
2. Navigate to R2 > Your bucket (e.g., "dataset")
3. Browse to the "mlfoundations-dclm-baseline-1.0-parquet" directory
4. Drag & drop to upload both `_shard_sizes.json` and `_metadata.yaml` files

This step is required to ensure your miner can properly access and use the dataset metadata.

### 11. Clear Local Cache (Required)

After uploading the metadata files, you must clear your local cache to force the miner to download the new metadata on its next run:

```bash
rm -rf ./.cache/tplr/*
```

## Troubleshooting

### Common Issues

1. **Download Interruption**: If your download is interrupted or times out, simply rerun the command. The downloader is designed to resume from where it left off, verifying existing files and continuing the process.

2. **Validation Failures**: If validation fails, check:
   - Your R2 bucket structure (should have "mlfoundations-dclm-baseline-1.0-parquet" subfolder)
   - Network connectivity during download
   - Cloudflare R2 permissions

3. **Validator Timeout**: If the validator times out, simply run it again. It's designed to resume from where it left off.
