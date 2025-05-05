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

### 2. Install and Run the Dataset Downloader

```bash
# Configure dataset bucket name to use
export DATABUCKET="dataset"

# Clone the downloader repository
git clone https://github.com/distributedstatemachine/HuggingFaceModelDownloader
cd HuggingFaceModelDownloader

# Create local .env file for R2 account credentials
tee .env << 'EOF'
R2_ACCOUNT_ID=your_account_id
R2_WRITE_ACCESS_KEY_ID=your_write_access_key
R2_WRITE_SECRET_ACCESS_KEY=your_write_secret_key
EOF

# Install Go (if not already installed)
wget https://go.dev/dl/go1.23.5.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.23.5.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# To make the PATH change permanent, add it to your .bashrc or .profile:
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc

# Verify Go installation
go version

# Start the download process
go run main.go \
  -d "mlfoundations/dclm-baseline-1.0-parquet" \
  --branch main \
  --hf-prefix "filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data" \
  --r2-subfolder "dclm-dataset" \
  --r2 --skip-local \
  --r2-bucket $DATABUCKET

# After the first download completes, run it again to verify and download any missing files
go run main.go \
  -d "mlfoundations/dclm-baseline-1.0-parquet" \
  --branch main \
  --hf-prefix "filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data" \
  --r2-subfolder "dclm-dataset" \
  --r2 --skip-local \
  --r2-bucket $DATABUCKET
```

![image](https://github.com/user-attachments/assets/f9235ac7-9861-4253-a7aa-24feca5e96ef)

![image](https://github.com/user-attachments/assets/a99737d1-259e-433d-9dcd-71e921c04e4c)

### 3. Validate the Dataset

After completing the download, validate that your uploaded dataset matches the expected file sizes and hashes:

```bash
# Clone the templar repository if you don't already have it
git clone git@github.com:tplr-ai/templar.git
cd templar/scripts/dclm-dataset-ops

# Run the validation script
./shards_validator.py "../../_shard_sizes.json" \
  --r2-bucket dataset \
  --r2-account-id $R2_ACCOUNT_ID \
  --r2-access-key-id $R2_READ_ACCESS_KEY_ID \
  --r2-access-key-secret $R2_READ_SECRET_ACCESS_KEY
```

![image](https://github.com/user-attachments/assets/0f68675f-d5b7-463b-b06e-318c1b0555c6)

This validates that all shards have been properly uploaded with the correct sizes and hashes. Review the validation results to ensure your dataset is complete and correct.

### 4. Configure Your Miner to Use the Dataset

Set these environment variables for your miner to connect to your Cloudflare R2 bucket:

```bash
export R2_DATASET_ACCOUNT_ID=...
export R2_DATASET_BUCKET_NAME=dataset
export R2_DATASET_READ_ACCESS_KEY_ID=...
export R2_DATASET_READ_SECRET_ACCESS_KEY=...
```

**Note**: For security, create separate read-only API keys for your miner. Never use your write access keys for the miner.

### 5. Clear Local Cache (if needed)

If you've run the miner previously with a different dataset configuration:

```bash
rm -rf ./.cache/tplr/*
```

## Troubleshooting

### Common Issues

1. **Download Interruption**: If your download is interrupted or times out, simply rerun the command. The downloader is designed to resume from where it left off, verifying existing files and continuing the process.

2. **Validation Failures**: If validation fails, check:
   - Your R2 bucket structure (should have "dclm-dataset" subfolder)
   - Network connectivity during download
   - Cloudflare R2 permissions

3. **Validator Timeout**: If the validator times out, simply run it again. It's designed to resume from where it left off.
