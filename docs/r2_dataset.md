# R2 Dataset Setup Overview

This document provides guidance on which dataset to use for Templar miners and links to the appropriate setup guides.

## Current Dataset: DCLM

**The DCLM (DataComp Language Model) dataset is the current dataset used by Templar miners.**

To set up the DCLM dataset, please follow the detailed instructions in the [DCLM Dataset Setup Guide](./r2_dataset-dclm.md).

### Quick Facts about DCLM

- Based on [mlfoundations/dclm-baseline-1.0-parquet](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet)
- Download time: 15-24 hours depending on network speed
- Required storage: 100GB for temporary processing
- Optimized with ZSTD compression and improved row group sizes

## Legacy Dataset: FineWeb-edu

The FineWeb-edu dataset was used in previous releases but is **no longer the current dataset**.

Setup instructions for FineWeb-edu are available in the [FineWeb-edu Dataset Setup Guide](./r2_dataset-fineweb-edu.md) for reference for legacy versions.

### Quick Facts about FineWeb-edu (Legacy)

- Based on [HuggingFaceFW/fineweb-edu-score-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2)
- Size: ~17.51 TB (as of 2/6/2025)
- Used in previous releases
- Setup process different from current DCLM dataset

## Important Notes

1. **Always use DCLM for current releases**: All miners should use the DCLM dataset for current versions.

2. **Legacy compatibility**: FineWeb-edu documentation is maintained for legacy releases.

3. **Bucket naming**: You can name your R2 bucket anything you like, but `dataset` is recommended for consistency.

4. **Security**: Always use separate read-only credentials for your miner and write credentials for dataset uploads.

## Getting Started

To begin setting up your dataset:

1. Choose the current dataset: [DCLM Dataset Setup Guide](./r2_dataset-dclm.md)
2. Follow the step-by-step instructions carefully
3. Pay special attention to using the correct downloader version
4. Ensure you have adequate bandwidth and storage

For any questions or issues, please contact the Templar team or open an issue on our GitHub repository.
