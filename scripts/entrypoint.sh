#!/bin/bash
set -e

# Check required environment variables
for var in WALLET_NAME WALLET_HOTKEY NODE_TYPE WANDB_API_KEY NETUID; do
    if [ -z "${!var}" ]; then
        echo "Error: $var environment variable is required"
        exit 1
    fi
done

# Activate virtual environment
source /app/.venv/bin/activate

# Create logs directory
mkdir -p /app/logs

# Check CUDA availability
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    echo "Error: CUDA is not available"
    exit 1
fi

# Login to wandb non-interactively
wandb login ${WANDB_API_KEY} --relogin

# Convert DEBUG to --debug flag if true
DEBUG_FLAG=""
if [ "$DEBUG" = "true" ]; then
    DEBUG_FLAG="--debug"
fi

# Extract GPU ID from CUDA_DEVICE (e.g., "cuda:0" -> "0")
if [[ "$CUDA_DEVICE" =~ cuda:([0-9]+) ]]; then
    GPU_ID="${BASH_REMATCH[1]}"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "Setting CUDA_VISIBLE_DEVICES=$GPU_ID"
else
    echo "Warning: Could not parse GPU ID from CUDA_DEVICE=$CUDA_DEVICE, using default"
fi

# Check CUDA version
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
if [[ "${CUDA_VERSION}" != "12.6" ]]; then
    echo "Warning: Container CUDA version (${CUDA_VERSION}) differs from host CUDA version (12.6)"
fi

# Check NODE_TYPE and start appropriate process
if [ "$NODE_TYPE" = "miner" ]; then
    echo "Starting miner with torchrun..."
    exec torchrun \
        --standalone \
        --nnodes 1 \
        --nproc_per_node 2 \
        neurons/miner.py \
        --wallet.name ${WALLET_NAME} \
        --wallet.hotkey ${WALLET_HOTKEY} \
        --netuid ${NETUID} \
        --device ${CUDA_DEVICE} \
        --subtensor.network ${NETWORK} \
        --use_wandb \
        ${PROJECT:+--project ${PROJECT}} \
        ${DEBUG_FLAG}
elif [ "$NODE_TYPE" = "validator" ]; then
    echo "Starting validator with torchrun..."
    exec torchrun \
        --standalone \
        --nnodes 1 \
        --nproc_per_node 1 \
        neurons/validator.py \
        --wallet.name ${WALLET_NAME} \
        --wallet.hotkey ${WALLET_HOTKEY} \
        --netuid ${NETUID} \
        --device ${CUDA_DEVICE} \
        --subtensor.network ${NETWORK} \
        --use_wandb \
        ${PROJECT:+--project ${PROJECT}} \
        ${DEBUG_FLAG}
else
    echo "Error: NODE_TYPE must be either \"miner\" or \"validator\""
    exit 1
fi 