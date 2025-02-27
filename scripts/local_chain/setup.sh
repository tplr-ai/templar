#!/bin/bash
# Setup script for local Subtensor chain
# Usage: ./setup.sh [--no-purge]

# Navigate to the script directory
cd "$(dirname "$0")"

# Create directories
mkdir -p data/alice data/bob chain-specs

# Check if we need to purge existing data
if [ "$1" != "--no-purge" ]; then
  echo "*** Purging previous state..."
  rm -rf data/alice/* data/bob/*
  echo "*** Previous chainstate purged"
else
  echo "*** Purging previous state skipped..."
fi

# Generate chain spec using a temporary container
echo "*** Building chainspec..."
docker run --rm -v "$(pwd)/chain-specs:/chain-specs" ghcr.io/opentensor/subtensor:v2.0.4 \
  /usr/local/bin/node-subtensor build-spec --disable-default-bootnode --raw --chain local > chain-specs/local.json
echo "*** Chainspec built and output to file"

# Generate node keys for Alice and Bob
echo "*** Generating node keys..."
docker run --rm -v "$(pwd)/data/alice:/data" ghcr.io/opentensor/subtensor:v2.0.4 \
  /usr/local/bin/node-subtensor key generate-node-key --file /data/node.key
docker run --rm -v "$(pwd)/data/bob:/data" ghcr.io/opentensor/subtensor:v2.0.4 \
  /usr/local/bin/node-subtensor key generate-node-key --file /data/node.key
echo "*** Node keys generated"

# Start the nodes
echo "*** Starting localnet nodes with docker-compose..."
docker compose up -d

# Wait for nodes to start
echo "*** Waiting for nodes to start..."
sleep 10

# Show logs to confirm startup
echo "*** Node logs:"
docker compose logs --tail=20

echo "*** Local Subtensor network is running!"
echo "Alice RPC endpoint: ws://localhost:9944"
echo "Bob RPC endpoint: ws://localhost:9945"