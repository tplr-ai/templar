# R2 Dataset Setup Overview

This document provides guidance on which dataset to use for Templar miners and links to the appropriate setup guides.

## Current Dataset: FineWeb-edu

**The FineWeb-edu dataset is the current dataset used by Templar miners.**

To set up the FineWeb-edu dataset, please follow the detailed instructions in the [FineWeb-edu Dataset Setup Guide](./r2_dataset-fineweb-edu.md).

### Quick Facts about FineWeb-edu:
- Based on [HuggingFaceFW/fineweb-edu-score-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2)
- Size: ~17.51 TB (as of 2/6/2025)
- Download time: 12-30 hours depending on network speed
- Required storage: 100GB for temporary processing

## Future Dataset: DCLM

The DCLM (DataComp Language Model) dataset is planned for future releases but is **not currently in use**.

Setup instructions for DCLM are available in the [DCLM Dataset Setup Guide](./r2_dataset-dclm.md) for reference, but miners should **not use this dataset yet**.

### Quick Facts about DCLM:
- Based on [mlfoundations/dclm-baseline-1.0-parquet](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet)
- Will be activated in a future release
- Setup process similar to FineWeb-edu but with different data

## Important Notes

1. **Always use FineWeb-edu for now**: Until officially announced, all miners should use the FineWeb-edu dataset.

2. **Dataset transitions**: When we transition to DCLM or other datasets, it will be announced through official channels.

3. **Bucket naming**: You can name your R2 bucket anything you like, but `dataset` is recommended for consistency.

4. **Security**: Always use separate read-only credentials for your miner and write credentials for dataset uploads.

## Getting Started

To begin setting up your dataset:

1. Choose the current dataset: [FineWeb-edu Dataset Setup Guide](./r2_dataset-fineweb-edu.md)
2. Follow the step-by-step instructions carefully
3. Pay special attention to using the correct downloader version
4. Ensure you have adequate bandwidth and storage

For any questions or issues, please contact the Templar team or open an issue on our GitHub repository.