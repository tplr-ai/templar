# Local Model Converter

The Local Model Converter is a standalone tool for converting Templar model checkpoints from local files to GGUF format and optionally uploading them to HuggingFace Hub and/or Ollama. Unlike the autonomous model converter service, this tool operates on local checkpoint files without requiring network access to the Bittensor network.

## Overview

This script provides a complete pipeline for converting your local model checkpoints to GGUF format and deploying them to popular model hosting services. It handles all the necessary dependencies and conversion steps automatically while preserving both the original HuggingFace format and the converted GGUF format. The tool can optionally upload models to HuggingFace Hub for sharing and to Ollama for local inference.

### TL;DR

To convert a local model checkpoint and upload it to HuggingFace and/or Ollama, run:

```bash
python scripts/model_converter-local.py \
  --checkpoint_path="./checkpoint-792063-1-v0.2.81.pt" \
  --device=cuda:1 --output_dir="./models" --version_tag="v0.2.81+792063" \
  --upload-hf --hf_token="..." --hf_repo_id=tplr/TEMPLAR-I
```

## Features

- **Local checkpoint loading**: Converts checkpoints from local file paths
- **Automatic versioning**: Generates semantic version tags from checkpoint metadata
- **GGUF conversion**: Creates GGUF files compatible with llama.cpp and other inference engines
- **Dual format preservation**: Always keeps both HuggingFace and GGUF formats
- **HuggingFace Hub upload**: Direct upload to HuggingFace repositories with authentication
- **Ollama integration**: Upload GGUF models directly to local Ollama installation
- **Flexible authentication**: Support for both environment variables and CLI token arguments
- **Repository management**: Automatic repository creation and configuration
- **Dependency management**: Automatically downloads required conversion scripts and packages
- **Detailed output**: Provides conversion statistics, upload results, and file locations
- **Dry run support**: Preview uploads without executing them

## Requirements

### System Dependencies

- Python 3.11+
- CUDA-capable GPU (recommended)
- Internet connection (for downloading conversion dependencies)
- Ollama (optional, for local model deployment)

### Python Dependencies

- torch
- transformers
- tplr (Templar package)
- gguf (automatically installed if missing)
- huggingface_hub (optional, for HuggingFace uploads)

## Installation

The script will automatically handle most dependencies. Ensure you have the Templar package installed:

```bash
uv pip install --pre -e ".[dev]"
```

For HuggingFace uploads, install the Hub library:

```bash
pip install huggingface_hub
```

For Ollama integration, install Ollama:

```bash
# On macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or visit https://ollama.ai for other installation methods
```

## Usage

### Basic Usage

Convert a checkpoint with default settings:

```bash
python scripts/model_converter-local.py --checkpoint_path model_checkpoint.pt
```

### Upload to HuggingFace Hub

Convert and upload to HuggingFace:

```bash
# Using environment variable for authentication
export HF_TOKEN=hf_your_token_here
python scripts/model_converter-local.py \
    --checkpoint_path model_checkpoint.pt \
    --upload-hf

# Using CLI token argument
python scripts/model_converter-local.py \
    --checkpoint_path model_checkpoint.pt \
    --upload-hf \
    --hf_token "hf_your_token_here"
```

### Upload to Ollama

Convert and upload to Ollama:

```bash
python scripts/model_converter-local.py \
    --checkpoint_path model_checkpoint.pt \
    --upload-ollama
```

### Upload to Both Services

Convert and upload to both HuggingFace and Ollama:

```bash
python scripts/model_converter-local.py \
    --checkpoint_path model_checkpoint.pt \
    --upload-hf \
    --upload-ollama \
    --hf_token "hf_your_token_here"
```

### Advanced Usage

Specify custom settings:

```bash
python scripts/model_converter-local.py \
    --checkpoint_path model_checkpoint.pt \
    --output_dir ./my_converted_models \
    --version_tag "v1.0.0-custom" \
    --device cuda:1 \
    --upload-hf \
    --hf_repo_id "myuser/my-templar-model" \
    --private \
    --commit_message "Initial model release"
```

### Dry Run

Preview what would be uploaded:

```bash
python scripts/model_converter-local.py \
    --checkpoint_path model_checkpoint.pt \
    --upload-hf \
    --upload-ollama \
    --dry_run
```

## Command Line Arguments

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint_path` | str | **Required** | Path to the model checkpoint file |
| `--device` | str | `"cuda"` | Device to use for model loading |
| `--output_dir` | str | `"converted_models"` | Directory to save converted models |
| `--version_tag` | str | `None` | Custom version tag (auto-generated if not provided) |

### Upload Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--upload-hf` | flag | `False` | Upload converted model to HuggingFace Hub |
| `--upload-ollama` | flag | `False` | Upload GGUF model to Ollama |
| `--hf_repo_id` | str | `None` | HuggingFace repository ID (auto-generated if not provided) |
| `--hf_token` | str | `None` | HuggingFace token for authentication (overrides HF_TOKEN env var) |
| `--ollama_model_name` | str | `None` | Custom Ollama model name (auto-generated if not provided) |
| `--private` | flag | `False` | Create private HuggingFace repository |
| `--commit_message` | str | `None` | Custom commit message for HuggingFace upload |
| `--dry_run` | flag | `False` | Show what would be uploaded without actually uploading |

## Version Tag Generation

If no custom version tag is provided, the script automatically generates one using the following priority:

1. **Window-based**: `{version}-local+{global_step}` (if checkpoint contains window information)
2. **Step-based**: `{version}-local+{global_step}` (if checkpoint contains global_step)
3. **Timestamp-based**: `{version}-local+{timestamp}` (fallback using current timestamp)

Where `{version}` is the current Templar package version.

## Output Structure

The script creates the following directory structure:

```
output_dir/
└── {version_tag}/
    ├── config.json              # Model configuration
    ├── model.safetensors         # HuggingFace model weights
    ├── model.gguf               # GGUF converted model
    ├── tokenizer.json           # Tokenizer configuration
    ├── tokenizer_config.json    # Tokenizer settings
    └── special_tokens_map.json  # Special token mappings
```

## Authentication Setup

### HuggingFace Hub

To upload to HuggingFace, you need a valid HuggingFace token:

1. **Get a token**: Visit [HuggingFace Settings](https://huggingface.co/settings/tokens) and create a new token with write permissions
2. **Set authentication** (choose one method):

```bash
# Method 1: Environment variable (recommended for regular use)
export HF_TOKEN=hf_your_token_here

# Method 2: CLI argument (recommended for one-off uploads)
python scripts/model_converter-local.py --hf_token "hf_your_token_here" --upload-hf

# Method 3: Login via CLI (alternative)
huggingface-cli login
```

### Ollama

Ensure Ollama is running:

```bash
# Start Ollama service
ollama serve

# Verify it's working
ollama list
```

## Conversion Process

The script follows these steps:

1. **Load Checkpoint**: Loads the model weights and metadata from the local checkpoint file
2. **Generate Version**: Creates a version tag based on metadata or custom input
3. **Save HuggingFace Format**: Saves the model in standard HuggingFace format
4. **Download Dependencies**: Ensures GGUF conversion script and packages are available
5. **Convert to GGUF**: Runs the llama.cpp conversion script to create GGUF format
6. **Upload to Services**: Optionally uploads to HuggingFace Hub and/or Ollama
7. **Report Results**: Displays conversion statistics, upload results, and file locations

## Output Example

```text
==================================================
CONVERSION & UPLOAD RESULTS
==================================================

Version: 0.2.88-local+1250
Model Directory: converted_models/0.2.88-local+1250
GGUF File: converted_models/0.2.88-local+1250/model.gguf
Total Time: 78.45 seconds
GGUF File Size: 2.85 GB

Uploads:
  ✅ HuggingFace: https://huggingface.co/templar-model-0-2-88-local-1250
  ✅ Ollama: templar-0.2.88-local+1250
     Test with: ollama run templar-0.2.88-local+1250
==================================================
```

## Troubleshooting

### Common Issues

**FileNotFoundError: Checkpoint not found**

- Verify the checkpoint path is correct
- Ensure you have read permissions for the file

**CUDA out of memory**

- Try using a different GPU device: `--device cuda:1`
- Use CPU if necessary: `--device cpu` (slower but works with limited GPU memory)

**GGUF conversion failed**

- Check that you have internet connection for downloading dependencies
- Ensure sufficient disk space for temporary files
- Verify that the checkpoint contains valid model weights

**HuggingFace authentication failed**

- Ensure `HF_TOKEN` is set or provide `--hf_token`
- Verify token has write permissions: visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
- Try logging in: `huggingface-cli login`

**HuggingFace upload failed**

- Check internet connection
- Verify repository name doesn't conflict with existing repos
- Ensure sufficient disk space for upload staging

**Ollama upload failed**

- Verify Ollama is installed and running: `ollama list`
- Check that Ollama service is accessible: `ollama serve`
- Ensure model name doesn't conflict: `ollama list`

**Import errors**

- Ensure Templar package is properly installed: `uv pip install --pre -e ".[dev]"`
- Install optional dependencies: `pip install huggingface_hub`
- Check that all dependencies are available in your environment

### Debug Information

The script provides detailed logging information. Check the console output for:

- Checkpoint loading progress
- Dependency installation status
- Conversion command details
- Upload progress and authentication status
- File size and location information

## Integration with Other Tools

The converted models can be used with:

### GGUF Format Compatibility

- **llama.cpp**: For CPU/GPU inference
- **Ollama**: For easy model serving (direct integration via `--upload-ollama`)
- **text-generation-webui**: For web-based interfaces
- **LM Studio**: For desktop model management

### HuggingFace Format Compatibility

- **Transformers**: Direct integration with HuggingFace ecosystem
- **vLLM**: High-performance inference serving
- **TensorRT-LLM**: NVIDIA-optimized inference
- **Text Generation Inference**: HuggingFace's production serving solution

## Performance Notes

- **GPU Usage**: Model loading requires GPU memory proportional to model size
- **Disk Space**: Conversion requires 2-3x model size in temporary disk space
- **Conversion Time**: Typically 30-60 seconds for standard model sizes
- **Upload Time**: Varies based on model size and internet speed
  - HuggingFace: 1-5 minutes for typical models
  - Ollama: Near-instantaneous (local operation)
- **File Preservation**: Both HuggingFace and GGUF formats are always preserved

## Related Documentation

- [Model Converter Service](model_converter.md) - For autonomous network-based conversion
- [Miner Setup](miner.md) - For checkpoint generation and training
- [Evaluator Local](evaluator-local.md) - For local model evaluation
