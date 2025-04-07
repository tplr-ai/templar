"""
Stop command implementation.
"""

import sys
from typing import List, Tuple
from rentcompute.config import Config
from rentcompute.providers.base import Pod


def run(config: Config, instance_id: str, skip_confirmation: bool = False) -> None:
    """Run the stop command (deprecated - use stop_instance instead).

    Args:
        config: Configuration manager
        instance_id: ID of the instance to stop
        skip_confirmation: Skip confirmation prompt if True
    """
    # Forward to new function for backwards compatibility
    stop_instance(config, instance_id, skip_confirmation)


def stop_instance(
    config: Config, instance_id: str, skip_confirmation: bool = False
) -> None:
    """Stop a specific instance by ID.

    Args:
        config: Configuration manager
        instance_id: ID of the instance to stop
        skip_confirmation: Skip confirmation prompt if True
    """
    # Get provider
    provider = config.get_provider()

    # Get active pods to verify the instance exists and get its name
    active_pods = provider.list_pods()
    found_pod = None

    # Check if the instance exists
    for pod in active_pods:
        if pod.id == instance_id:
            found_pod = pod
            break

    if not found_pod:
        print(f"Warning: Instance {instance_id} not found among active instances.")
        if not skip_confirmation:
            confirm = input("Do you want to try stopping it anyway? (y/N): ")
            if confirm.lower() != "y":
                print("Operation cancelled.")
                return
    else:
        print(f"Found instance: {found_pod.name} (ID: {instance_id})")
        print(f"  Host: {found_pod.host}")
        print(f"  Status: {found_pod.status}")
        print(f"  GPU: {found_pod.gpu_count}x {found_pod.gpu_type}")

        if not skip_confirmation:
            confirm = input("Are you sure you want to stop this instance? (y/N): ")
            if confirm.lower() != "y":
                print("Operation cancelled.")
                return

    # Try to stop the instance using the provider
    print(f"Stopping instance {instance_id}...")

    try:
        success, error_message = provider.stop_pod(instance_id)
        if success:
            print(f"Instance {instance_id} stopped successfully.")
        else:
            _handle_stop_error(instance_id, error_message)
    except Exception as e:
        print(f"Error stopping instance: {e}")
        sys.exit(1)


def stop_all(config: Config, skip_confirmation: bool = False) -> None:
    """Stop all active instances.

    Args:
        config: Configuration manager
        skip_confirmation: Skip confirmation prompt if True
    """
    # Get provider
    provider = config.get_provider()

    # Get active pods
    active_pods = provider.list_pods()

    if not active_pods:
        print("No active instances found.")
        return

    # Display instances to be stopped
    print(f"Found {len(active_pods)} active instances:")
    for i, pod in enumerate(active_pods, 1):
        print(f"{i}. {pod.name} (ID: {pod.id})")
        print(f"   Host: {pod.host}")
        print(f"   Status: {pod.status}")
        print(f"   GPU: {pod.gpu_count}x {pod.gpu_type}")
        print(f"   Hourly rate: ${pod.hourly_rate:.2f}/hr")
        if i < len(active_pods):
            print()  # Empty line between instances

    # Ask for confirmation unless skipped
    if not skip_confirmation:
        confirm = input(
            f"\nAre you sure you want to stop ALL {len(active_pods)} instances? (y/N): "
        )
        if confirm.lower() != "y":
            print("Operation cancelled.")
            return

    # Stop all instances
    print(f"\nStopping {len(active_pods)} instances...")

    results: List[Tuple[Pod, bool, str]] = []  # (pod, success, error_message)

    for pod in active_pods:
        print(f"Stopping {pod.name} (ID: {pod.id})...")
        try:
            success, error_message = provider.stop_pod(pod.id)
            results.append((pod, success, error_message))
        except Exception as e:
            results.append((pod, False, str(e)))

    # Display results
    print("\nStop operation results:")
    success_count = sum(1 for _, success, _ in results if success)

    for pod, success, error_message in results:
        if success:
            print(f"✓ {pod.name} (ID: {pod.id}): Stopped successfully")
        else:
            print(f"✗ {pod.name} (ID: {pod.id}): Failed - {error_message}")

    print(f"\nSuccessfully stopped {success_count} of {len(active_pods)} instances.")

    if success_count < len(active_pods):
        print(
            "Some instances failed to stop. You may want to try again or stop them individually."
        )


def _handle_stop_error(instance_id: str, error_message: str) -> None:
    """Handle error messages from stopping an instance.

    Args:
        instance_id: ID of the instance that failed to stop
        error_message: Error message from the provider
    """
    # Check if the error indicates that the pod doesn't exist
    if error_message and (
        "Pod for executor" in error_message and "doesn't exist" in error_message
    ):
        print(
            f"Instance {instance_id} is not currently running or has already been stopped."
        )
    # Check if the error is related to invalid UUID format
    elif error_message and "Input should be a valid UUID" in error_message:
        print(
            f"Error: '{instance_id}' is not a valid instance ID. Instance IDs must be valid UUIDs."
        )
    # Check if the executor doesn't exist
    elif (
        error_message
        and "Executor" in error_message
        and "doesn't exist" in error_message
    ):
        print(f"Instance {instance_id} does not exist or has already been terminated.")
    else:
        print(
            f"Failed to stop instance {instance_id}. Please check the instance ID and try again."
        )
        print(f"Error: {error_message}")
    sys.exit(1)
