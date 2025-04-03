"""
Provision command implementation.
"""

from typing import Dict, Any, List, Optional
import logging

from rentcompute.config import Config
from rentcompute.provisioning import provision_instance

logger = logging.getLogger(__name__)


def run(config: Config, instance_id: str, skip_confirmation: bool = False) -> None:
    """Run the provision command to provision an existing instance.

    Args:
        config: Configuration manager
        instance_id: ID of the instance to provision
        skip_confirmation: Whether to skip the confirmation prompt
    """
    # Get the provider from config
    provider = config.get_provider()

    print(f"Looking up instance with ID: {instance_id}...")

    # List all pods and find the one with the matching ID
    pods = provider.list_pods()
    target_pod = None

    for pod in pods:
        if pod.id == instance_id:
            target_pod = pod
            break

    if not target_pod:
        print(f"Error: No active instance found with ID {instance_id}")
        print("Use 'rentcompute list' to see active instances")
        return

    print(f"Found instance '{target_pod.name}' (ID: {target_pod.id})")
    print(f"Host: {target_pod.host}, Port: {target_pod.port}")
    print(f"GPU: {target_pod.gpu_count}x {target_pod.gpu_type}")
    print(f"Hourly rate: ${target_pod.hourly_rate:.2f}/hr")

    # Ask for confirmation unless skipped
    if not skip_confirmation:
        confirm = input("Provision this instance? (y/n): ")
        if confirm.lower() != "y":
            print("Operation cancelled.")
            return
    else:
        print("Skipping confirmation due to -y/--yes flag")

    # Provision the instance
    print("\nProvisioning requested. Looking for .rentcompute.yml in current directory...")
    if provision_instance(target_pod):
        print("Provisioning completed successfully.")
        
        # Reprint SSH connection details
        private_key_path = (
            target_pod.key_path.replace(".pub", "")
            if target_pod.key_path.endswith(".pub")
            else target_pod.key_path
        )
        print("\nSSH Connection Details:")
        print(f"Host: {target_pod.host}")
        print(f"User: {target_pod.user}")
        print(f"Port: {target_pod.port}")
        print(f"SSH Key: {private_key_path}")
        print(f"\nConnection command:")
        print(f"ssh {target_pod.user}@{target_pod.host} -p {target_pod.port} -i {private_key_path}")
    else:
        print("Provisioning failed. Instance is still running but may require manual setup.")