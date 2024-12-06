
Clone:
```bash
# Clone package
git clone git@github.com:unconst/tplr.git
cd tplr
```

Install requirements:
```bash
# Install uv.
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
# Install npm.
curl -L https://www.npmjs.com/install.sh | sh
# Install pm2.
npm install -g pm2
```

Set up the environment:
```bash
# Create uv venv
uv venv
# Activate venv
source .venv/bin/activate
# Install reqs.
uv sync
```

Run a node (miner or validator):
```bash
# Run script for miner and validator.
pm2 start run.py \
    # run.py interpreter.
    --interpreter python3 \
    # pm2 process name
    --name node 
    # Switch to run args.
    -- \
    # Walelt specific coldkey to use.
    --wallet.name ... \
    # wallet specific hotkey to use.
    --wallet.hotkey ... \
    # Bittensor network UID.
    --netuid ... \ 
    # Train with specific peer uids from the network only e.g., --peers 1 2 3.
    --peers ... \
    # Device to use for training (e.g., cpu or cuda).
    --device ... \ 
    # Enable debug logging.
    --debug \
    # Enable trace logging.
    --trace \
    # Use Weights and Biases for logging.
    --use_wandb \
    # Evaluate other miners and set weights on the chain.
    --is_validator \
    # Trains on a random page instead of correctly assigned. (for testing).
    --random \
```