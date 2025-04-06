"""
Rsync command implementation.
"""

import os
import subprocess
import yaml
import logging
from typing import Dict, List, Optional, Tuple

from rentcompute.config import Config

logger = logging.getLogger(__name__)

# Standard rsync exclusions for development projects
STANDARD_EXCLUSIONS = [
    "--exclude=node_modules",
    "--exclude=target",
    "--exclude=venv",
    "--exclude=.venv",
    "--exclude=dist",
    "--exclude=build",
]


def run(
    config: Config,
    instance_id: Optional[str] = None,
    config_path: str = ".rentcompute.yml",
    skip_confirmation: bool = False,
    reload_after: bool = False,
) -> None:
    """Run the rsync command to sync directories with compute instances.

    Args:
        config: Configuration manager
        instance_id: Optional ID of a specific instance to sync with
        config_path: Path to the configuration file
        skip_confirmation: Whether to skip the confirmation prompt
        reload_after: Whether to reload instances after sync
    """
    # Get the provider from config
    provider = config.get_provider()

    # Load sync configuration from .rentcompute.yml
    sync_config = load_sync_config(config_path)
    if not sync_config:
        print(f"No sync configuration found in {config_path}")
        print("Please add a 'sync' section with source and destination directories")
        return

    # Get active pods
    active_pods = provider.list_pods()
    if not active_pods:
        print("No active instances found.")
        return

    # Filter pods by instance_id if specified
    target_pods = []
    if instance_id:
        for pod in active_pods:
            if pod.id == instance_id:
                target_pods.append(pod)
                break
        if not target_pods:
            print(f"No active instance found with ID {instance_id}")
            return
    else:
        target_pods = active_pods

    # Display sync plan
    print(f"Will sync directories with {len(target_pods)} instance(s):")
    for i, pod in enumerate(target_pods, 1):
        print(f"{i}. {pod.name} (ID: {pod.id})")
        print(f"   Host: {pod.host}")
        print(f"   User: {pod.user}")
        print(f"   Port: {pod.port}")
        if i < len(target_pods):
            print()

    print("\nDirectories to sync:")
    for i, sync_item in enumerate(sync_config, 1):
        source = os.path.abspath(os.path.expanduser(sync_item["source"]))
        destination = sync_item["destination"]
        print(f"{i}. Local: {source} -> Remote: {destination}")

    # Ask for confirmation unless skipped
    if not skip_confirmation:
        confirm = input("\nStart syncing directories? (y/N): ")
        if confirm.lower() != "y":
            print("Operation cancelled.")
            return
    else:
        print("\nSkipping confirmation due to -y/--yes flag")

    # Perform rsync for each pod and sync item
    results = []
    for pod in target_pods:
        pod_results = []
        print(f"\nSyncing with {pod.name} (ID: {pod.id})...")

        # Get the SSH key path (private key, not .pub)
        private_key_path = (
            pod.key_path.replace(".pub", "")
            if pod.key_path.endswith(".pub")
            else pod.key_path
        )

        for sync_item in sync_config:
            source = os.path.abspath(os.path.expanduser(sync_item["source"]))
            destination = sync_item["destination"]

            # Verify source exists
            if not os.path.exists(source):
                print(f"Warning: Source path does not exist: {source}")
                pod_results.append((sync_item, False, "Source path does not exist"))
                continue

            # Run rsync command
            success, error = rsync_directories(
                source, destination, pod.host, pod.user, pod.port, private_key_path
            )

            pod_results.append((sync_item, success, error))

        results.append((pod, pod_results))

    # Print summary
    print("\nSync results:")
    for pod, pod_results in results:
        print(f"\n{pod.name} (ID: {pod.id}):")
        success_count = sum(1 for _, success, _ in pod_results if success)
        print(f"  {success_count}/{len(pod_results)} directories synced successfully")

        for sync_item, success, error in pod_results:
            source = os.path.abspath(os.path.expanduser(sync_item["source"]))
            destination = sync_item["destination"]

            if success:
                print(f"  ✓ {source} -> {destination}")
            else:
                print(f"  ✗ {source} -> {destination}: {error}")

    # Reload instances if requested
    if reload_after:
        print("\n=== Reloading instances after sync ===")

        # Import reload module here to avoid circular imports
        from rentcompute.commands import reload as reload_cmd

        # Load reload configuration
        reload_config = reload_cmd.load_reload_config(config_path)
        if not reload_config:
            print(f"No reload configuration found in {config_path}")
            print("Please add a 'reload' section to enable automatic reload after sync")
            return

        # Reload each instance that was synced
        reload_success_count = 0
        for pod, _ in results:
            print(f"\nReloading {pod.name} (ID: {pod.id})...")
            if reload_cmd.reload_pod(pod, reload_config):
                reload_success_count += 1

        # Print reload summary
        print(
            f"\nReload summary: {reload_success_count}/{len(results)} instances reloaded successfully"
        )


def load_sync_config(config_path: str) -> List[Dict[str, str]]:
    """Load sync configuration from the specified file.

    Args:
        config_path: Path to the configuration file

    Returns:
        List of sync items with source and destination
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return []

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Extract sync configuration
        if config and "sync" in config:
            return config["sync"]
        return []
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return []


def rsync_directories(
    source: str,
    destination: str,
    host: str,
    user: str,
    port: int,
    private_key_path: str,
) -> Tuple[bool, str]:
    """Sync directories using rsync.

    Args:
        source: Local source directory
        destination: Remote destination directory
        host: Remote host
        user: Remote user
        port: SSH port
        private_key_path: Path to SSH private key

    Returns:
        Tuple of (success, error_message)
    """
    # Ensure source ends with / to sync contents, not the directory itself
    if not source.endswith("/"):
        source = f"{source}/"

    # Build rsync command
    rsync_cmd = [
        "rsync",
        "-avzP",  # archive, verbose, compress, show progress
        "--delete",  # delete extraneous files on destination
    ]

    # Add standard exclusions
    rsync_cmd.extend(STANDARD_EXCLUSIONS)

    # Add SSH options
    ssh_options = f"ssh -p {port} -i {private_key_path} -o StrictHostKeyChecking=no"
    rsync_cmd.extend(["-e", ssh_options])

    # Add source and destination
    rsync_cmd.append(source)
    rsync_cmd.append(f"{user}@{host}:{destination}")

    print(f"Syncing: {source} -> {user}@{host}:{destination}")

    try:
        # Execute rsync command
        process = subprocess.Popen(
            rsync_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Stream output line by line
        for line in iter(process.stdout.readline, ""):
            print(f"  {line.rstrip()}")

        # Wait for process to complete
        process.stdout.close()
        return_code = process.wait()

        if return_code == 0:
            return True, ""
        else:
            return False, f"rsync exited with code {return_code}"
    except Exception as e:
        return False, str(e)
