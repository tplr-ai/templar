"""
List instances command implementation.
"""

from typing import List
from rentcompute.config import Config
from rentcompute.providers.base import Pod


def run(config: Config) -> None:
    """Run the list command.

    Args:
        config: Configuration manager
    """
    try:
        # Get provider for active pod data
        provider = config.get_provider()
        active_pods = provider.list_pods()

        # Show active pods from provider
        if active_pods:
            print(f"Active instances ({len(active_pods)}):")
            _print_pods_table(active_pods)
        else:
            print("No active instances found.")
    except Exception as e:
        print(f"Could not fetch instances from provider: {e}")
        print("No active instances found.")


def _print_pods_table(pods: List[Pod]) -> None:
    """Print a formatted table of pods.

    Args:
        pods: List of Pod objects to display
    """
    # Header
    print(
        "Name                 | ID                                   | Host                   | User     | Port | Status   | GPU Type                  | GPU Count | Price ($/hr) | SSH Command"
    )
    print("-" * 150)

    # Sort pods by name
    sorted_pods = sorted(pods, key=lambda p: p.name)

    for pod in sorted_pods:
        # Format status with color (green for running, yellow for other states)
        status_display = (
            f"\033[92m{pod.status}\033[0m"
            if pod.status.lower() == "running"
            else f"\033[93m{pod.status}\033[0m"
        )

        # Format hourly rate
        price_display = f"${pod.hourly_rate:.2f}"

        # Create SSH command
        # For SSH connection, we need the private key (without .pub extension)
        key_path = (
            pod.key_path.replace(".pub", "")
            if pod.key_path.endswith(".pub")
            else pod.key_path
        )
        ssh_command = f"ssh {pod.user}@{pod.host} -p {pod.port} -i {key_path}"

        # Truncate long values
        name = pod.name[:20] if len(pod.name) > 20 else pod.name
        id_str = pod.id[:36] if len(pod.id) > 36 else pod.id
        host = pod.host[:23] if len(pod.host) > 23 else pod.host
        gpu_type = pod.gpu_type[:25] if len(pod.gpu_type) > 25 else pod.gpu_type

        print(
            f"{name:<20} | {id_str:<36} | {host:<23} | {pod.user:<8} | {pod.port:<4} | {status_display:<8} | "
            f"{gpu_type:<25} | {pod.gpu_count:<9} | {price_display:<12} | {ssh_command}"
        )
