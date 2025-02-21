# S3 Bucket Manager for Cloudflare R2

A Python utility to manage objects in Cloudflare R2 buckets with various deletion capabilities.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/downloads/release/python-3129/)
[![Cloudflare R2](https://img.shields.io/badge/Cloudflare-R2-orange)](https://cloudflare.com/products/r2/)

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Configuration Options](#configuration-options)
5. [Examples](#examples)
6. [Error Handling](#error-handling)
7. [Security Considerations](#security-considerations)

---

## Features

- Delete objects older than a specified number of hours
- Delete objects matching a specific prefix pattern
- Delete objects ending with a specified suffix
- Wipe entire bucket (use with caution!)
- Flexible configuration options for different buckets
- Compatible with Cloudflare R2 storage
- Logging and error handling
- Confirmation prompt for critical operations

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tplr-ai/templar.git
   cd templar
   ```

2. Install required Python packages:
   ```bash
   pip install uv
   uv python install 3.12
   uv python pin 3.12
   uv venv .venv
   source .venv/bin/activate 
   uv pip install -e . --prerelease=allow
   uv pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

3. Set up your Cloudflare R2 API keys (see [Security Considerations](#security-considerations)).

---

## Usage

The script allows you to manage objects in an S3-compatible bucket using the following actions:

```bash
python ./scripts/s3_manager.py [--action] [--config-options]
```

### Actions (Choose One or More)

| Action                     | Description                                      |
|----------------------------|--------------------------------------------------|
| `--delete-old X`           | Delete objects older than X hours                 |
| `--prefix PREFIX`          | Delete objects starting with specified prefix    |
| `--suffix SUFFIX`          | Delete objects ending with specified suffix      |
| `--wipe-bucket`            | Delete ALL objects in the bucket (DANGEROUS)     |

### Configuration Options

| Option                     | Description                                      | Required?           |
|----------------------------|--------------------------------------------------|---------------------|
| `--miner-bucket`           | Use R2_Gradients* environment variables         | If using gradients  |
| `--dataset-bucket`         | Use R2_Dataset* environment variables            | If using dataset    |
| `--bucket-name NAME`       | Specify custom bucket name                      | If not using presets|
| `--access-key KEY`         | Custom access key                               | For custom config  |
| `--secret-key KEY`         | Custom secret key                               | For custom config  |
| `--account-id ID`          | Cloudflare R2 account ID                        | For custom config  |
| `--endpoint-url URL`       | Custom S3-compatible endpoint URL               | Optional           |

---

## Examples

### Basic Usage
```bash
# Delete all objects older than 12 hours in the default bucket
python s3_manager.py --delete-old 12 --miner-bucket

# Wipe entire bucket (with confirmation)
python s3_manager.py --wipe-bucket --dataset-bucket

# Delete objects with prefix "backup"
python s3_manager.py --prefix "gradient" --bucket-name "my_gradient_bucket" \
--access-key "MY_KEY" --secret-key "MY_SECRET" --account-id "MY_ACCOUNT_ID"
```

### Custom Configuration
```bash
# Delete objects ending with ".csv" in a custom bucket
python s3_manager.py --suffix ".csv" --bucket-name "data_lake" \
--access-key "CUSTOM_KEY" --secret-key "CUSTOM_SECRET" \
--endpoint-url "https://my-r2-bucket.example.com"
```

### Full Custom Setup
```bash
# Delete all objects older than 24 hours with custom configuration
python s3_manager.py --delete-old 24 --bucket-name "logs_bucket" \
--access-key "LOGS_KEY" --secret-key "LOGS_SECRET" \
--account-id "CLOUDFLARE_ID"
```

---

## Error Handling

The script includes error handling for common issues:

| Error                          | Solution                                      |
|---------------------------------|------------------------------------------------|
| `RequestLimitExceeded`         | Wait and try again                            |
| `NoSuchBucket`                 | Verify bucket name                            |
| `InvalidAccessKeyId`           | Check your access key                         |
| `SignatureDoesNotMatch`        | Verify secret key                             |

---

## Security Considerations

1. **Permissions**:
   - Ensure your R2 API key has minimal required permissions (write)
   - Use separate keys for different buckets

2. **Wipe Bucket**:
   - Be extremely careful with `--wipe-bucket`
   - Make sure you're operating on the correct bucket

---

## License

[MIT License](LICENSE)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.