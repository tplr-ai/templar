# Load environment variables
source .env_test

# pm2 delete all
pm2 start neurons/miner.py --interpreter python3 --name TM0 -- --wallet.name ${WALLET_NAME} --wallet.hotkey ${WALLET_HOTKEY}_M1 --device ${CUDA_DEVICE} --subtensor.network ${NETWORK} --debug --netuid ${NETUID} --project ${1:-default}

pm2 start neurons/miner.py --interpreter python3 --name TM1 -- --wallet.name ${WALLET_NAME} --wallet.hotkey ${WALLET_HOTKEY}_M2 --device ${CUDA_DEVICE} --subtensor.network ${NETWORK} --use_wandb --debug --netuid ${NETUID} --project ${1:-default}

pm2 start neurons/validator.py --interpreter python3 --name TV1 -- --wallet.name ${WALLET_NAME} --wallet.hotkey ${WALLET_HOTKEY} --device ${CUDA_DEVICE} --subtensor.network ${NETWORK} --use_wandb --debug --netuid ${NETUID} --project ${1:-default}
