# The MIT License (MIT)
# © 2025 tplr.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import bittensor as bt
import tplr


def add_neuron_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments used by both miner and validator."""
    parser.add_argument(
        "--netuid", type=int, default=268, help="Bittensor network UID."
    )
    parser.add_argument(
        "--project", type=str, default="templar", help="Wandb project."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--trace", action="store_true", help="Enable trace logging")
    parser.add_argument(
        "--store-gathers",
        action="store_true",
        help="Store gathered gradients in R2",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - use all peers without filtering",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Local run - use toy model, small enough for a laptop.",
    )
    
    # Add Bittensor arguments
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)


def get_config(neuron_type: str) -> bt.Config:
    """
    Create and configure a bt.Config object for the specified neuron type.
    
    Args:
        neuron_type: Either "miner" or "validator"
        
    Returns:
        Configured bt.Config object
    """
    parser = argparse.ArgumentParser(description=f"{neuron_type.capitalize()} script")
    
    # Add common neuron arguments
    add_neuron_args(parser)
    
    # Add neuron-specific arguments
    if neuron_type == "miner":
        parser.add_argument(
            "--actual-batch-size",
            type=int,
            default=None,
            help="Override the batch size defined in hparams.",
        )
    # No validator-specific args currently needed
    
    # Create config
    config = bt.config(parser)
    
    # Apply debug/trace logging levels
    if config.debug:
        tplr.debug()
    if config.trace:
        tplr.trace()
    
    return config 