#!/usr/bin/env python3
# Setup subnet and fund wallets for local Subtensor chain

import time
import argparse
import sys
import bittensor as bt
import substrateinterface


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Set up local subtensor subnet")

    # Add standard bittensor wallet arguments
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)

    # Add validator and miner hotkey arguments
    parser.add_argument(
        "--validator.hotkey", type=str, required=True, help="Validator hotkey name"
    )

    parser.add_argument(
        "--miner.hotkeys", type=str, nargs="+", default=[], help="Miner hotkey names"
    )

    # Configuration
    parser.add_argument(
        "--stake.amount",
        type=float,
        default=10.0,
        help="Amount to stake for the validator (in TAO)",
    )
    parser.add_argument(
        "--fund.amount",
        type=float,
        default=100.0,
        help="Amount to fund each wallet (in TAO)",
    )

    # Parse args
    config = bt.config(parser)

    # IMPORTANT: Force connection to local network
    config.subtensor.network = "ws://localhost:9944"
    config.subtensor.chain_endpoint = "ws://localhost:9944"
    # Connect to local subtensor
    print(f"Connecting to subtensor at {config.subtensor.network}")
    subtensor = bt.subtensor(config=config)

    try:
        # Access //Alice account directly using keypair
        print("Accessing //Alice dev account...")
        alice_keypair = substrateinterface.Keypair.create_from_uri("//Alice")
        alice_address = alice_keypair.ss58_address
        alice_balance = subtensor.get_balance(alice_address)
        print(f"Dev account (//Alice) balance: {alice_balance}")

        # Check if we have enough funds
        total_needed = config.fund.amount * (1 + len(config.miner.hotkeys or []))
        if float(alice_balance) < total_needed:
            print(
                f"ERROR: Dev account doesn't have enough funds. Balance: {alice_balance}, Needed: {total_needed}"
            )
            sys.exit(1)

        # Get wallet object
        wallet = bt.wallet(config=config)
        coldkey_address = wallet.coldkeypub.ss58_address

        # Fund wallet coldkey
        print(f"Funding wallet coldkey {coldkey_address} with {config.fund.amount} TAO")
        subtensor.transfer(
            alice_address, coldkey_address, config.fund.amount, alice_keypair
        )

        # Wait for transaction to be processed
        print("Waiting for fund transfer to be processed...")
        time.sleep(3)

        # Register a subnet using Alice keypair
        print("Registering new subnet using dev account...")
        subnet_result = subtensor.create_subnet(alice_keypair)
        if subnet_result is not True:
            print(f"ERROR: Failed to create subnet: {subnet_result}")
            sys.exit(1)

        # Verify subnet creation - we expect it to be netuid 2
        subnets = subtensor.get_subnets()
        print(f"Available subnets: {subnets}")

        if 2 not in subnets:
            print(f"ERROR: Expected netuid 2 not found in available subnets: {subnets}")
            sys.exit(1)

        # Wait a moment for subnet registration to propagate
        print("Waiting for subnet registration to propagate...")
        time.sleep(2)

        # Set validator hotkey and stake to register its neuron
        print(f"Setting validator hotkey: {config.validator.hotkey}")
        validator_wallet = bt.wallet(
            name=config.wallet.name, hotkey=config.validator.hotkey
        )

        print(f"Staking {config.stake.amount} TAO for validator...")
        stake_result = subtensor.add_stake(
            validator_wallet,
            validator_wallet.hotkey.ss58_address,
            2,  # netuid 2
            config.stake.amount,
        )
        print(f"Validator stake result: {stake_result}")

        # Verify validator neuron registration
        metagraph = subtensor.metagraph(2)
        hotkeys = [uid.hotkey for uid in metagraph.neurons]
        if validator_wallet.hotkey.ss58_address in hotkeys:
            print(
                f"Validator neuron successfully registered with hotkey: {validator_wallet.hotkey.ss58_address}"
            )
        else:
            print(
                f"WARNING: Validator hotkey {validator_wallet.hotkey.ss58_address} not found in metagraph"
            )

        print("\nLocal subtensor subnet setup complete!")
        print(f"Chain endpoint: {config.subtensor.network}")
        print("Subnet ID: 2")
        print(f"Wallet: {config.wallet.name}")
        print(f"Validator hotkey: {config.validator.hotkey}")
        print(f"Miner hotkeys: {config.miner.hotkeys}")

    except Exception as e:
        print(f"ERROR: Failed during setup: {e}")
        raise


if __name__ == "__main__":
    main()
