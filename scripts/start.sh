#!/bin/bash
# Load environment variables
source .env

# Stop any existing processes
pm2 delete all

# # Generate random suffix for project name
RANDOM_SUFFIX=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 4 | head -n 1)
PROJECT_NAME="test_${RANDOM_SUFFIX}"

# Start miners and validator with matching configurations
pm2 start neurons/miner.py --interpreter python3 --name TM1 -- \
  --wallet.name Bistro \
  --wallet.hotkey M111 \
  --device cuda:3 \
  --subtensor.network test \
  --debug \
  --netuid 268 \
  --use_wandb \
  --project "${PROJECT_NAME}"

pm2 start neurons/miner.py --interpreter python3 --name TM2 -- \
  --wallet.name Bistro \
  --wallet.hotkey M222 \
  --device cuda:1 \
  --subtensor.network test \
  --debug \
  --netuid 268 \
  --use_wandb \
  --project "${PROJECT_NAME}"

pm2 start neurons/validator.py --interpreter python3 --name TV1 -- \
  --wallet.name Bistro \
  --wallet.hotkey V11 \
  --device cuda:2 \
  --subtensor.network test \
  --debug \
  --netuid 268 \
  --use_wandb \
  --project "${PROJECT_NAME}"

