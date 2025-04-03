#!/usr/bin/env python3
"""
CLI interface for rentcompute
"""

import sys
import logging
import argparse
from typing import List, Optional

from rentcompute import __version__
from rentcompute.commands import login, start, list_instances, stop, search, provision, rsync, reload
from rentcompute.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

# Create logger for this module
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="rentcompute", description="Rent compute instances easily"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Login command
    subparsers.add_parser("login", help="Set API credentials")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start a new compute instance")
    
    # Provision command
    provision_parser = subparsers.add_parser("provision", help="Provision an existing compute instance")
    provision_parser.add_argument(
        "--id", 
        dest="instance_id",
        required=True,
        help="ID of the instance to provision"
    )
    provision_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    start_parser.add_argument(
        "--name",
        dest="name",
        type=str,
        help="Custom name for the instance",
        default=None,
    )
    # Direct machine ID selection
    start_parser.add_argument(
        "--id",
        dest="machine_id",
        type=str,
        help="Specific machine ID to start (bypasses all filtering)",
        default=None,
    )
    # GPU parameters
    start_parser.add_argument(
        "--gpu-min",
        dest="gpu_min",
        type=int,
        help="Minimum number of GPUs",
        default=None,
    )
    start_parser.add_argument(
        "--gpu-max",
        dest="gpu_max",
        type=int,
        help="Maximum number of GPUs",
        default=None,
    )
    start_parser.add_argument(
        "--gpu-type",
        dest="gpu_type",
        type=str,
        help="Type of GPU (e.g., h100)",
        default=None,
    )
    # Price parameters
    start_parser.add_argument(
        "--price-min",
        dest="price_min",
        type=float,
        help="Minimum price per hour in USD",
        default=None,
    )
    start_parser.add_argument(
        "--price-max",
        dest="price_max",
        type=float,
        help="Maximum price per hour in USD",
        default=None,
    )
    # SSH key parameter
    start_parser.add_argument(
        "--ssh-key",
        dest="ssh_key",
        type=str,
        help="Path to SSH public key file (default: ~/.ssh/id_rsa.pub)",
        default="~/.ssh/id_rsa.pub",
    )
    # Provisioning option
    start_parser.add_argument(
        "--provision",
        dest="provision",
        action="store_true",
        help="Provision the instance using configuration from .rentcompute.yml",
    )

    # Search command - same arguments as start but only shows availability
    search_parser = subparsers.add_parser(
        "search", help="Search for available compute instances without starting them"
    )
    search_parser.add_argument(
        "--name",
        dest="name",
        type=str,
        help="Filter results by name pattern",
        default=None,
    )
    # GPU parameters
    search_parser.add_argument(
        "--gpu-min",
        dest="gpu_min",
        type=int,
        help="Minimum number of GPUs",
        default=None,
    )
    search_parser.add_argument(
        "--gpu-max",
        dest="gpu_max",
        type=int,
        help="Maximum number of GPUs",
        default=None,
    )
    search_parser.add_argument(
        "--gpu-type",
        dest="gpu_type",
        type=str,
        help="Type of GPU (e.g., h100)",
        default=None,
    )
    # Price parameters
    search_parser.add_argument(
        "--price-min",
        dest="price_min",
        type=float,
        help="Minimum price per hour in USD",
        default=None,
    )
    search_parser.add_argument(
        "--price-max",
        dest="price_max",
        type=float,
        help="Maximum price per hour in USD",
        default=None,
    )

    # List command
    subparsers.add_parser("list", help="List active compute instances")

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop a compute instance")
    stop_group = stop_parser.add_mutually_exclusive_group(required=True)
    stop_group.add_argument("--all", action="store_true", help="Stop all active instances")
    stop_group.add_argument("--id", dest="instance_id", help="ID of a specific instance to stop")
    stop_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    
    # Rsync command
    rsync_parser = subparsers.add_parser("rsync", help="Sync directories with compute instances")
    rsync_group = rsync_parser.add_mutually_exclusive_group()
    rsync_group.add_argument("--all", action="store_true", help="Sync with all active instances", default=True)
    rsync_group.add_argument("--id", dest="instance_id", help="ID of a specific instance to sync with")
    rsync_parser.add_argument(
        "--config",
        dest="config_path",
        default=".rentcompute.yml",
        help="Path to the configuration file (default: .rentcompute.yml)",
    )
    rsync_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    rsync_parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload instances after sync",
    )
    
    # Reload command
    reload_parser = subparsers.add_parser("reload", help="Reload running compute instances")
    reload_group = reload_parser.add_mutually_exclusive_group(required=True)
    reload_group.add_argument("--all", action="store_true", help="Reload all active instances")
    reload_group.add_argument("--id", dest="instance_id", help="ID of a specific instance to reload")
    reload_parser.add_argument(
        "--config",
        dest="config_path",
        default=".rentcompute.yml",
        help="Path to the configuration file (default: .rentcompute.yml)",
    )
    reload_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        int: Exit code
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = create_parser()
    args = parser.parse_args(argv)

    # Set log level based on debug flag
    if hasattr(args, "debug") and args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    if not args.command:
        parser.print_help()
        return 1

    config = Config()

    try:
        if args.command == "login":
            login.run(config)
            return 0

        # Check if credentials exist for all other commands
        try:
            # Just check if we can load credentials, we don't need to use them here
            config.load_credentials()
        except FileNotFoundError:
            print("No credentials found. Please enter your API key:")
            login.run(config)

        # Continue with the requested command
        if args.command == "start":
            # Combine all filter parameters
            instance_config = {
                "machine_id": args.machine_id,  # Direct machine ID selection
                "gpu": {
                    "min": args.gpu_min,
                    "max": args.gpu_max,
                    "type": args.gpu_type,
                },
                "price": {
                    "min": args.price_min,
                    "max": args.price_max,
                },
                "ssh_key": args.ssh_key,
                "provision": args.provision,  # Add provision flag
            }
            start.run(config, instance_config, name=args.name)
        elif args.command == "search":
            # Combine all filter parameters
            instance_config = {
                "gpu": {
                    "min": args.gpu_min,
                    "max": args.gpu_max,
                    "type": args.gpu_type,
                },
                "price": {
                    "min": args.price_min,
                    "max": args.price_max,
                },
            }
            search.run(config, instance_config, name=args.name)
        elif args.command == "list":
            list_instances.run(config)
        elif args.command == "stop":
            if args.all:
                stop.stop_all(config, skip_confirmation=args.yes)
            else:
                stop.stop_instance(config, args.instance_id, skip_confirmation=args.yes)
        elif args.command == "provision":
            provision.run(config, args.instance_id, skip_confirmation=args.yes)
        elif args.command == "rsync":
            # Determine if we're using --all or a specific instance
            instance_id = None if args.all or not args.instance_id else args.instance_id
            rsync.run(config, instance_id=instance_id, config_path=args.config_path, 
                     skip_confirmation=args.yes, reload_after=args.reload)
        elif args.command == "reload":
            # Determine if we're using --all or a specific instance
            if args.all:
                reload.reload_all(config, config_path=args.config_path, skip_confirmation=args.yes)
            else:
                reload.reload_instance(config, args.instance_id, config_path=args.config_path, 
                                     skip_confirmation=args.yes)
        else:
            parser.print_help()
            return 1

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
