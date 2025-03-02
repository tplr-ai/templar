# Local Subtensor Chain (WIP - Do not use)

This directory contains scripts for running a local Subtensor blockchain with Docker. It allows you to quickly set up a development environment for bittensor-based projects.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.8+

## Quick Start

1. **Start the local chain:**
   ```bash
   cd scripts/local_chain
   ./setup.sh
   ```

2. **Create a subnet and fund your wallet:**
   ```bash
   ./setup_subnet.py \
     --wallet.name YourWallet \
     --validator.hotkey validator \
     --miner.hotkeys miner1 miner2
   ```

3. **Run your validators and miners against the local chain:**
   ```bash
   python neurons/validator.py \
     --wallet.name YourWallet \
     --wallet.hotkey validator \
     --subtensor.network ws://localhost:9944 \
     --netuid 2

   python neurons/miner.py \
     --wallet.name YourWallet \
     --wallet.hotkey miner1 \
     --subtensor.network ws://localhost:9944 \
     --netuid 2
   ```

## Detailed Usage

### Setting Up the Chain

The `setup.sh` script:
- Creates necessary directories
- Purges previous chain state (unless `--no-purge` is specified)
- Generates a local chain specification
- Starts two validator nodes (Alice and Bob) using Docker Compose
- Exposes RPC endpoints at localhost:9944 and localhost:9945

Options:
- `--no-purge`: Keeps existing chain data (useful for resuming a previous session)

Example:
```bash
./setup.sh --no-purge
```

### Creating a Subnet and Funding Wallets

The `setup_subnet.py` script:
- Connects to your local chain
- Funds your wallet from the dev account (//Alice)
- Creates a new subnet (which becomes netuid 2)
- Registers the specified validator hotkey on the subnet
- Verifies successful registration

Required arguments:
- `--wallet.name`: Name of your wallet
- `--validator.hotkey`: Hotkey to register as a validator

Optional arguments:
- `--miner.hotkeys`: Space-separated list of miner hotkeys
- `--stake.amount`: Amount to stake for the validator (default: 10.0 TAO)
- `--fund.amount`: Amount to fund your wallet (default: 100.0 TAO)

Example with custom funding:
```bash
./setup_subnet.py \
  --wallet.name YourWallet \
  --validator.hotkey validator \
  --miner.hotkeys miner1 miner2 \
  --fund.amount 500.0 \
  --stake.amount 20.0
```

## Directory Structure

- `data/` - Contains the blockchain data for Alice and Bob nodes
- `chain-specs/` - Contains the generated chain specification
- `setup.sh` - Script to initialize and start the local chain
- `setup_subnet.py` - Script to create a subnet and fund wallets
- `docker-compose.yml` - Docker Compose configuration for the local chain

## Troubleshooting

### Chain doesn't start properly

Check Docker logs for errors:
```bash
docker-compose logs
```

### Connection issues

Ensure the RPC ports are correctly exposed:
```bash
docker ps
```

You should see ports 9944 and 9945 mapped to your host.

### Funding or subnet creation fails

- Ensure your local chain is running (`docker ps` should show the containers)
- Check your wallet exists (`btcli wallet list`)
- Try running with the `--subtensor.network ws://localhost:9944` flag explicitly

### Clean restart

To completely reset the chain:
```bash
docker-compose down
rm -rf data chain-specs
./setup.sh
```

## Notes

- The local chain is pre-funded with development accounts (//Alice, //Bob)
- Each time you run `setup.sh` without `--no-purge`, you'll get a fresh chain
- Subnet ID is expected to be 2 for the first created subnet
- All TAO on the local chain has no real-world value
