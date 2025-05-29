# FineWeb-edu Dataset Setup for Miners (Legacy)

This guide explains how to properly download, validate, and set up the FineWeb-edu dataset for use with your miner. **This dataset is for legacy releases only.**

> **Important**: This dataset is no longer the current dataset. For current releases, use the [DCLM Dataset Setup Guide](./r2_dataset-dclm.md).

## Dataset Information

The FineWeb-edu dataset is based on the [HuggingFaceFW/fineweb-edu-score-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2) dataset on Hugging Face. This was the primary dataset used by Templar miners in previous releases.

> **Note**: The DCLM dataset is now the current dataset for active releases.

## System Requirements

You will need to make sure you have adequate bandwidth. **17.51 TB of data** (as of 2/6/2025) will be transferred through your machine but NOT be stored locally by default. The higher the internet speed and more workers/cores you can utilize, the faster the dataset will be processed. Consider that this process will take 12-30+ hours, so use screen/tmux accordingly.

### Recommended Hardware

- **Network**: 1gbps+
- **Local Storage**: 100GB (for temporary processing)  
- **RAM**: 4GB+ (this process is not memory intensive)
- **CPU Cores**: 8+ (more cores = faster processing)
- **Estimated Download Time**: 12-30 hours (depending on network speed)

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

# IMPORTANT: Checkout the specific commit for fineweb-edu dataset
git checkout c72fcc0ef5fccf3b1c4fcf39cf7ccda0bad8093d
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

> **Note**: The `-c` flag specifies the number of concurrent connections. More cores = faster transfer.

```bash
# Start the dataset transfer
go run main.go \
  -d "HuggingFaceFW/fineweb-edu-score-2" \
  --r2 --skip-local -c $CPUCOUNT \
  --branch v1.2.0 \
  --r2-bucket $DATABUCKET
```

### 7. Check for Corrupted Files

After the initial transfer, check for any corrupted files:

```bash
go run main.go \
  -d "HuggingFaceFW/fineweb-edu-score-2" \
  --r2 --cleanup-corrupted \
  --branch v1.2.0 \
  --r2-bucket $DATABUCKET
```

### 8. Verify and Re-run Transfer

If the process was interrupted or you want to ensure all files are present, run the transfer again:

```bash
# Re-run to catch any missed files
go run main.go \
  -d "HuggingFaceFW/fineweb-edu-score-2" \
  --r2 --skip-local -c $CPUCOUNT \
  --r2-bucket $DATABUCKET
```

The downloader is designed to resume from where it left off, verifying existing files and continuing the process.

### 9. Update Metadata Files

After the transfer is complete, return to the templar repository to configure metadata:

```bash
# Return to the templar repo
cd ~/templar

# Update the _shard_sizes.json to use your bucket name
sed -i 's|80f15715bb0b882c9e967c13e677ed7d|'"$DATABUCKET"'|g' _shard_sizes.json

# Clear any local cache from previous runs
rm -rf ./.cache/tplr/*
```

### 10. Configure Your Miner to Use the Dataset

Set these environment variables for your miner to connect to your Cloudflare R2 bucket:

```bash
export R2_DATASET_ACCOUNT_ID=$R2_ACCOUNT_ID
export R2_DATASET_BUCKET_NAME=$DATABUCKET
export R2_DATASET_READ_ACCESS_KEY_ID=your_read_access_key
export R2_DATASET_READ_SECRET_ACCESS_KEY=your_read_secret_key
```

**Note**: For security, create separate read-only API keys for your miner. Never use your write access keys for the miner.

## Troubleshooting

### Common Issues

1. **Download Interruption**: If your download is interrupted, simply rerun the transfer command. The downloader will verify existing files and continue from where it left off.

2. **Wrong Dataset**: If you accidentally used the main branch, you may be downloading the wrong dataset. Make sure to checkout the specific commit: `c72fcc0ef5fccf3b1c4fcf39cf7ccda0bad8093d`

3. **Slow Transfer**: The transfer speed depends on your internet connection and CPU cores. Use more cores (`-c` flag) for faster transfers if available.

4. **Storage Issues**: While the data isn't stored locally by default, ensure you have at least 100GB free for temporary processing.

## Additional Resources

- For questions or issues, please contact the Templar team or open an issue on our GitHub repository
- For dataset overview and options, see the [R2 Dataset Overview](./r2_dataset.md)
- For alternative dataset setup (future use), see the [DCLM Dataset Guide](./r2_dataset-dclm.md)
