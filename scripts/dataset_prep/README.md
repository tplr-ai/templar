# Data Preparation Pipeline

This directory contains scripts for preparing training data for Templar. The pipeline pretokenizes 150B tokens data into manageable shards and then consolidates them for efficient dataloading.

## Overview

The data preparation process consists of two main steps:

1. **Pretokenization** (`01_pretokenize_data.py`): Downloads and tokenizes raw text data, saving it as numpy shards
2. **Consolidation** (`02_consolidate_shards.py`): Combines shards into optimized binary files for training

## Quick Start

### Option 1: Pretokenize from scratch

```bash
# Set your Hugging Face token for faster/more stable downloads
export HF_TOKEN=your_token_here

# Run pretokenization (creates ~150 shards of 1B tokens each)
python 01_pretokenize_data.py --output_dir <DATA_ROOT>
```

### Option 2: Download pretokenized data

```bash
# Configure R2 access
export AWS_ACCESS_KEY_ID=a796d0b0f5601b4e6e5d55c445bc4088
export AWS_SECRET_ACCESS_KEY=ae4ffa21d265fd928e2f47a809ca96efcc56a1a59d40979fa5081d9c6dfadf57
ENDPOINT="https://c17ae32eab2883094481526a22e0dfa1.r2.cloudflarestorage.com"

# Install AWS CLI and download
pip install awscli
aws s3 --endpoint-url "$ENDPOINT" sync "s3://pretokenized-dataset/" <DATA_ROOT>
```

### Step 2: Consolidate shards

After obtaining the pretokenized shards, consolidate them for efficient training:

```bash
python 02_consolidate_shards.py --data_root <DATA_ROOT>
```

This creates:
- `tokens.bin`: All tokens in a single memory-mapped file
- `sample_ids.bin`: Unique identifiers for each training sample

## Dataset Details

- **Source**: DCLM baseline dataset (mlfoundations/dclm-baseline-1.0-parquet)
- **Total tokens**: 150G tokens (150 * (1024 ** 3))
- **Shard size**: 1B tokens per shard (150 shards)
- **Tokenizer**: LLaMA-2-7B-32K (togethercomputer/LLaMA-2-7B-32K)
- **Sequence length**: 2048 tokens
- **Token dtype**: uint16

## Output Structure

After running both scripts, your data directory will contain:

```
<DATA_ROOT>/
├── train_000000.npy          # Original shards (can be deleted after consolidation)
├── train_000001.npy
├── ...
├── train_000149.npy
├── tokens.bin                # Consolidated tokens (memory-mapped)
└── sample_ids.bin          
```

## Notes

- **Pretokenization**: Uses multiprocessing (default: 75% of CPU cores). Takes ~2-4 hours.
- **Cloud Download**: Takes ~15-30 minutes
- **Storage after download**: ~300 GB
- **Storage after consolidation**: ~600 GB


**Validation**: The consolidation script includes SHA-256 verification against expected checksums to ensure data integrity. Make sure to check the logs!

**💡 Tip**: After consolidation, look for the "✅ Preprocessing complete!" message indicating all SHA-256 checksums passed. Once validated, you can safely delete all `train_*.npy` shards to free up ~300GB of disk space:
```bash
rm <DATA_ROOT>/train_*.npy
```
Only `tokens.bin` and `sample_ids.bin` are needed for training.