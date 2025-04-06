"""
Reload command implementation.
"""

import os
import subprocess
import yaml
import logging
from typing import Dict, Any
from pathlib import Path

from rentcompute.config import Config
from rentcompute.providers.base import Pod

logger = logging.getLogger(__name__)


def reload_all(
    config: Config,
    config_path: str = ".rentcompute.yml",
    skip_confirmation: bool = False,
) -> bool:
    """Reload all active instances.

    Args:
        config: Configuration manager
        config_path: Path to the configuration file
        skip_confirmation: Whether to skip the confirmation prompt

    Returns:
        True if all reloads were successful, False otherwise
    """
    # Get provider
    provider = config.get_provider()

    # Get active pods
    active_pods = provider.list_pods()
    if not active_pods:
        print("No active instances found.")
        return False

    # Load reload configuration
    reload_config = load_reload_config(config_path)
    if not reload_config:
        print(f"No reload configuration found in {config_path}")
        print("Please add a 'reload' section with the required configuration")
        return False

    # Display instances to be reloaded
    print(f"Found {len(active_pods)} active instances:")
    for i, pod in enumerate(active_pods, 1):
        print(f"{i}. {pod.name} (ID: {pod.id})")
        print(f"   Host: {pod.host}")
        print(f"   Status: {pod.status}")
        if i < len(active_pods):
            print()

    # Ask for confirmation unless skipped
    if not skip_confirmation:
        confirm = input(f"\nReload ALL {len(active_pods)} instances? (y/N): ")
        if confirm.lower() != "y":
            print("Operation cancelled.")
            return False
    else:
        print(
            f"Skipping confirmation due to -y/--yes flag. Reloading {len(active_pods)} instances..."
        )

    # Reload all instances
    success_count = 0
    for pod in active_pods:
        print(f"\nReloading {pod.name} (ID: {pod.id})...")
        success = reload_pod(pod, reload_config)
        if success:
            success_count += 1

    # Print summary
    print(f"\nSuccessfully reloaded {success_count} of {len(active_pods)} instances.")

    return success_count == len(active_pods)


def reload_instance(
    config: Config,
    instance_id: str,
    config_path: str = ".rentcompute.yml",
    skip_confirmation: bool = False,
) -> bool:
    """Reload a specific instance.

    Args:
        config: Configuration manager
        instance_id: ID of the instance to reload
        config_path: Path to the configuration file
        skip_confirmation: Whether to skip the confirmation prompt

    Returns:
        True if reload was successful, False otherwise
    """
    # Get provider
    provider = config.get_provider()

    # Load reload configuration
    reload_config = load_reload_config(config_path)
    if not reload_config:
        print(f"No reload configuration found in {config_path}")
        print("Please add a 'reload' section with the required configuration")
        return False

    # Get active pods to verify the instance exists
    active_pods = provider.list_pods()
    found_pod = None

    # Check if the instance exists
    for pod in active_pods:
        if pod.id == instance_id:
            found_pod = pod
            break

    if not found_pod:
        print(f"No active instance found with ID {instance_id}")
        print("Use 'rentcompute list' to see active instances")
        return False

    # Display instance details
    print(f"Found instance: {found_pod.name} (ID: {instance_id})")
    print(f"  Host: {found_pod.host}")
    print(f"  Status: {found_pod.status}")
    print(f"  GPU: {found_pod.gpu_count}x {found_pod.gpu_type}")

    # Ask for confirmation unless skipped
    if not skip_confirmation:
        confirm = input("\nReload this instance? (y/N): ")
        if confirm.lower() != "y":
            print("Operation cancelled.")
            return False
    else:
        print("Skipping confirmation due to -y/--yes flag")

    # Reload the instance
    return reload_pod(found_pod, reload_config)


def reload_pod(pod: Pod, reload_config: Dict[str, Any]) -> bool:
    """Reload a pod using the specified configuration.

    Args:
        pod: The pod to reload
        reload_config: Reload configuration from .rentcompute.yml

    Returns:
        True if reload was successful, False otherwise
    """
    print(f"Reloading instance {pod.name} (ID: {pod.id})...")

    # Check reload type
    reload_type = reload_config.get("type", "").lower()

    if reload_type == "ansible":
        return _reload_with_ansible(pod, reload_config)
    else:
        print(f"Unsupported reload type: {reload_type}")
        print("Currently only 'ansible' reload type is supported")
        return False


def _reload_with_ansible(pod: Pod, config: Dict[str, Any]) -> bool:
    """Reload instance using Ansible.

    Args:
        pod: Instance pod to reload
        config: Reload configuration

    Returns:
        True if reload was successful, False otherwise
    """
    playbook_path = config.get("playbook")
    if not playbook_path:
        print("No playbook specified in reload configuration")
        return False

    # Check for root_dir parameter and switch to that directory if specified
    current_dir = os.getcwd()
    root_dir = config.get("root_dir")
    ansible_dir = current_dir

    if root_dir:
        # Resolve to absolute path using Path.resolve() which handles .., ., and ~
        try:
            # First expand any ~ in the path
            expanded_path = os.path.expanduser(root_dir)

            # Create a Path object - if relative, it will be relative to current_dir
            root_path = Path(expanded_path)

            # Resolve to absolute path (handles .. and . correctly)
            resolved_path = root_path.resolve()

            # Check if directory exists
            if resolved_path.is_dir():
                ansible_dir = str(resolved_path)
                print(f"Using Ansible root directory: {ansible_dir}")

                # Convert playbook path to be relative to the ansible_dir if it's not absolute
                if not os.path.isabs(playbook_path):
                    playbook_path = os.path.join(ansible_dir, playbook_path)
            else:
                print(
                    f"Warning: Specified Ansible root directory '{resolved_path}' not found. Using current directory."
                )
        except Exception as e:
            print(
                f"Error resolving Ansible root directory: {e}. Using current directory."
            )

    # Check if playbook exists
    if not os.path.exists(playbook_path):
        print(f"Ansible playbook not found: {playbook_path}")
        return False

    # Create temporary inventory file in ansible_dir
    inventory_path = Path(ansible_dir) / ".rentcompute_reload_inventory.ini"

    private_key_path = (
        pod.key_path.replace(".pub", "")
        if pod.key_path.endswith(".pub")
        else pod.key_path
    )

    try:
        # Get the target hosts group from config or use default
        hosts_group = config.get("hosts_group", "rentcompute")

        with open(inventory_path, "w") as f:
            f.write(f"[{hosts_group}]\n")
            f.write(
                f"{pod.host} ansible_user={pod.user} ansible_port={pod.port} ansible_ssh_private_key_file={private_key_path}\n"
            )

        # Initialize extra_vars_args list
        extra_vars_args = []

        # Check for vars_file if specified
        vars_file = config.get("vars_file")
        if vars_file:
            # Calculate the path relative to ansible_dir if it's not absolute
            if not os.path.isabs(vars_file):
                vars_file_path = os.path.join(ansible_dir, vars_file)
            else:
                vars_file_path = vars_file

            # Verify the vars file exists
            if os.path.isfile(vars_file_path):
                print(f"Using vars file: {vars_file_path}")
                extra_vars_args.extend(["-e", f"@{vars_file_path}"])
            else:
                print(f"Warning: Vars file not found: {vars_file_path}")

        # Get inline extra vars if specified
        extra_vars = config.get("extra_vars", {})
        if extra_vars:
            for key, value in extra_vars.items():
                extra_vars_args.extend(["-e", f"{key}={value}"])

        # Build ansible command with increased verbosity for better debugging
        cmd = ["ansible-playbook", "-i", str(inventory_path), playbook_path, "-v"]
        cmd.extend(extra_vars_args)

        # Change working directory if needed for ansible execution
        original_dir = os.getcwd()
        try:
            if ansible_dir != original_dir:
                os.chdir(ansible_dir)
                print(f"Changed to directory: {ansible_dir}")

            # Create environment with ANSIBLE_HOST_KEY_CHECKING=false
            env = os.environ.copy()
            env["ANSIBLE_HOST_KEY_CHECKING"] = "false"

            print(f"Running Ansible reload with playbook: {playbook_path}")

            # Run Ansible with live output streaming
            print("\n--- Ansible Reload Output ---")
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Stream output in real-time
            for line in iter(process.stdout.readline, ""):
                print(line, end="")  # Print line by line as they come in

            # Wait for process to complete and get return code
            process.stdout.close()
            return_code = process.wait()
            print("--- End of Ansible Output ---\n")

            if return_code == 0:
                print("Reload completed successfully.")

                # Reprint SSH connection details
                print("\nSSH Connection Details:")
                print(f"Host: {pod.host}")
                print(f"User: {pod.user}")
                print(f"Port: {pod.port}")
                print(f"SSH Key: {private_key_path}")
                print("\nConnection command:")
                print(f"ssh {pod.user}@{pod.host} -p {pod.port} -i {private_key_path}")

                return True
            else:
                print(f"Reload failed with exit code {return_code}")
                return False
        finally:
            # Return to original directory
            if ansible_dir != original_dir:
                os.chdir(original_dir)
    except Exception as e:
        print(f"Error during Ansible reload: {e}")
        return False
    finally:
        # Clean up temporary inventory file
        if inventory_path.exists():
            inventory_path.unlink()


def load_reload_config(config_path: str) -> Dict[str, Any]:
    """Load reload configuration from the specified file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Reload configuration dictionary
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return {}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Extract reload configuration
        if config and isinstance(config, dict) and "reload" in config:
            return config["reload"]
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}
