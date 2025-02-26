#!/usr/bin/env python3
# Setup subnet and fund wallets for local Subtensor chain

import time
import bittensor as bt
import argparse

def main():
    parser = argparse.ArgumentParser(description="Set up local subtensor subnet and fund wallets")
    
    # Admin wallet
    parser.add_argument("--wallet.name", type=str, required=True, help="Admin wallet name")
    parser.add_argument("--wallet.hotkey", type=str, required=True, help="Admin wallet hotkey or seed")
    
    # Validator wallet
    parser.add_argument("--validator.wallet.name", type=str, required=True, help="Validator wallet name")
    parser.add_argument("--validator.wallet.hotkey", type=str, required=True, help="Validator wallet hotkey or seed")
    
    # Miner wallets (optional)
    parser.add_argument("--miner.wallet.names", type=str, nargs='+', default=[], help="Miner wallet names")
    parser.add_argument("--miner.wallet.hotkeys", type=str, nargs='+', default=[], help="Miner wallet hotkeys or seeds")
    
    # Configuration
    parser.add_argument("--subtensor.chain_endpoint", type=str, default="ws://localhost:9944",
                       help="Subtensor chain endpoint")
    parser.add_argument("--stake.amount", type=float, default=10.0, 
                       help="Amount to stake for the validator (in TAO)")
    parser.add_argument("--fund.amount", type=float, default=100.0, 
                       help="Amount to fund each wallet (in TAO)")
    
    # Parse args
    config = bt.config(parser)
    
    # Wait for chain to be ready
    print(f"Connecting to subtensor at {config.subtensor.chain_endpoint}")
    
    # Initialize subtensor connection
    subtensor = bt.subtensor(config=config)
    
    # Initialize wallets
    admin_wallet = bt.wallet(name=config.wallet.name, hotkey=config.wallet.hotkey)
    validator_wallet = bt.wallet(name=config.validator.wallet.name, hotkey=config.validator.wallet.hotkey)
    
    miner_wallets = []
    if hasattr(config, 'miner') and hasattr(config.miner, 'wallet'):
        for name, hotkey in zip(config.miner.wallet.names, config.miner.wallet.hotkeys):
            miner_wallets.append(bt.wallet(name=name, hotkey=hotkey))
    
    # Check admin wallet balance
    admin_balance = subtensor.get_balance(admin_wallet.coldkeypub.ss58_address)
    print(f"Admin wallet balance: {admin_balance}")
    
    # Fund validator wallet
    print(f"Funding validator wallet {validator_wallet.hotkey.ss58_address} with {config.fund.amount} TAO")
    subtensor.transfer(
        admin_wallet.coldkeypub.ss58_address, 
        validator_wallet.hotkey.ss58_address, 
        config.fund.amount, 
        admin_wallet.coldkey
    )
    
    # Fund miner wallets
    for i, miner_wallet in enumerate(miner_wallets):
        print(f"Funding miner wallet {i+1}: {miner_wallet.hotkey.ss58_address} with {config.fund.amount} TAO")
        subtensor.transfer(
            admin_wallet.coldkeypub.ss58_address, 
            miner_wallet.hotkey.ss58_address, 
            config.fund.amount, 
            admin_wallet.coldkey
        )
    
    # Wait for transactions to be processed
    print("Waiting for fund transfers to be processed...")
    time.sleep(3)
    
    # Register a subnet
    print("Registering new subnet...")
    subnet_result = subtensor.create_subnet(admin_wallet)
    if subnet_result is not True:
        raise RuntimeError(f"Failed to create subnet: {subnet_result}")
    
    # Verify subnet creation - we expect it to be netuid 2
    subnets = subtensor.get_subnets()
    print(f"Available subnets: {subnets}")
    
    # Wait a moment for subnet registration to propagate
    time.sleep(2)
    
    # Stake the validator to register its neuron
    print(f"Staking {config.stake.amount} TAO for validator...")
    stake_result = subtensor.add_stake(
        validator_wallet,
        validator_wallet.hotkey.ss58_address,
        2,  # netuid 2
        config.stake.amount
    )
    print(f"Validator stake result: {stake_result}")
    
    # Verify validator neuron registration
    metagraph = subtensor.metagraph(2)
    hotkeys = [uid.hotkey for uid in metagraph.neurons]
    if validator_wallet.hotkey.ss58_address in hotkeys:
        print(f"Validator neuron successfully registered with hotkey: {validator_wallet.hotkey.ss58_address}")
    else:
        print(f"Warning: Validator hotkey {validator_wallet.hotkey.ss58_address} not found in metagraph")
    
    print("\nLocal subtensor subnet setup complete!")
    print(f"Chain endpoint: {config.subtensor.chain_endpoint}")
    print(f"Subnet ID: 2")

if __name__ == "__main__":
    main()