# Shared Sharded Dataset (2 trillion tokens)

This guide explains how to access the shards for the 2T tokens dataset. 

As we scale up to larger models, we need more data for the hungry model to consume. For the next run we'll have 6x more data, with about 2TB of storage required for everything on disk. At the largest model sizes, we'll need even more than that! 

With our miners and validators in mind, we wanted to avoid disk storage pressure but maintain the advantages of having the data locally. To do that, we will be blocking the download of the first ~200GB shard (S1). We're seeing this should take about 20m under the current codebase defaults, assuming ~150 cpus. The next shard will download in the background (S2). 

At the end of the allocated windows, the `SharedShardedDatasetManager` will swap the datasets and the dataloader will be refreshed. The upcoming shard (S3) will automatically downloaded in the background. And so on through the end of the training run (S4:). 

In order to keep the disk utilization skinny, shards are deleted after their respective window is over. If the model does another pass over the data, the initial shard is redownloaded and the process continues. To avoid shard deletion on 2TB+ local storage disks, see the final section on [Self-Hosting](#self-hosting).

## Dataset Information

The Shared Sharded dataset is based on the [mlfoundations/dclm-baseline-1.0-parquet](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet) dataset on Hugging Face, part of the research available at [DataComp](https://data.commoncrawl.org/contrib/datacomp/index.html).

For the fastest training, our optimized version includes:

- Pretokenized numpy arrays in .npy files
- Array slicing provided via .bin files

## System Requirements

The dataset transfer requires adequate bandwidth. Files are temporarily processed on your machine but not permanently stored locally. Higher internet speeds and more worker threads will significantly accelerate the process.

Expect to simultaneously hold 2+ dataset shards at one time. Each shard is ~215GB. As one shard downloads, another is downloading. When a swap happens, both will be on disk and another one may be downloading to the tmp directory.

### Recommended Hardware

- **Network**: 1gbps+
- **Local Storage**: 750GB (for temporary processing)
- **RAM**: 4GB+
- **CPU Cores**: 100+ (more cores = faster downloading the datasets)
- **Estimated Download Time**: 20 minutes per shard (depending on network speed and cpus)

## Setup Instructions

### Modify your Cloudflare R2 `dataset` Bucket settings

Append the following env keys:
```bash
R2_DATASET_ACCOUNT_ID=
R2_DATASET_BUCKET_NAME=dataset
R2_DATASET_READ_ACCESS_KEY_ID= 
R2_DATASET_READ_SECRET_ACCESS_KEY=
DATASET_BINS_PATH="remote/tokenized/"
```

## Troubleshooting

### Common Issues

1. **Network Interruptions**: The comms module, called in the `SharedShardedDatasetManager`, will automatically retry 

2. **Slow Downloads**: Consider:
   - Edit `comms.py` in the `download_large_file` function and hard-code the `max_workers=total_chunks`
   - Using a machine with higher network bandwidth
   - Using a machine with more CPUs

3. **Space Issues**: If you run out of space during temporary file processing, ensure you have at least 225GB of free space or use a machine with more storage.

4. **HTTP Timeout Errors**: These are handled automatically with retries. If you see persistent timeouts, there might be network connectivity issues to either Hugging Face or Cloudflare.

## Performance Considerations

- **Optimizing Download Speed**: If you were to decrease the `chunk_size` to something like 2 * 1024**3, you could use 100 workers in parallel 
- **Memory Usage**: Not sure but use plenty of RAM if you modify anything
- **Storage I/O**: Using SSD storage for temporary files will significantly improve performance

## Self-Hosting <a name="self-hosting"></a>

There are two useful directives when self hosting: 
* Self bucket management with the CloudFlare migration tool 
* Preventing `os.remove` during the dataset swap

### R2 Self-Hosting
For the self hosting, please use the CloudFlare migration tool and ...
{needs filled in}

### Keeping inactive shards on disk
The `SharedShardedDatasetManager` class does handle object deletion for inactive shards. 

To prevent this:
* Navigate to `src/tplr/sharded_dataset` 
* Comment out the loop starting with `if old_dataset and self.rank == 0:` with `#` at the line beginning

Since that loop (with `os.remove`) would have deleted the inactive shards, now they will stay on disk. 

The total disk utilization of all shards is > 2TB. 