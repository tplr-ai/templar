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

# Check CUDA version
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
if [[ "${CUDA_VERSION}" != "12.6" ]]; then
    echo "Warning: Container CUDA version (${CUDA_VERSION}) differs from host CUDA version (12.6)"
fi

# Check NODE_TYPE and start appropriate process
if [ "$NODE_TYPE" = "miner" ]; then
    echo "Starting miner..."
    exec python3 neurons/miner.py \
        --wallet.name ${WALLET_NAME} \
        --wallet.hotkey ${WALLET_HOTKEY} \
        --netuid ${NETUID} \
        --device ${CUDA_DEVICE} \
        --subtensor.network ${NETWORK} \
        --use_wandb \
        ${DEBUG_FLAG}
elif [ "$NODE_TYPE" = "validator" ]; then
    if [ "$RUN_AGGREGATOR" = "true" ] && [ "$RUN_EVALUATOR" = "true" ]; then
        echo "Running aggregator and evaluator concurrently..."
        echo "Starting aggregator on cuda:1..."
        python3 neurons/aggregator.py \
            --wallet.name ${WALLET_NAME} \
            --wallet.hotkey ${WALLET_HOTKEY} \
            --netuid ${NETUID} \
            --device "cuda:1" \
            --subtensor.network ${NETWORK} \
            --use_wandb \
            ${DEBUG_FLAG} &
        aggregator_pid=$!
        echo "Starting evaluator on cuda:2..."
        python3 scripts/evaluator.py \
            --netuid ${NETUID} \
            --device "cuda:2" \
            --use_wandb \
            ${DEBUG_FLAG} &
        evaluator_pid=$!
        wait $aggregator_pid $evaluator_pid
    else
        echo "Starting validator..."
        exec python3 neurons/validator.py \
            --wallet.name ${WALLET_NAME} \
            --wallet.hotkey ${WALLET_HOTKEY} \
            --netuid ${NETUID} \
            --device ${CUDA_DEVICE} \
            --subtensor.network ${NETWORK} \
            --use_wandb \
            ${DEBUG_FLAG}
    fi
else
    echo "Error: NODE_TYPE must be either \"miner\" or \"validator\""
    exit 1
fi 