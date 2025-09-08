#!/usr/bin/env python3
import asyncio
import subprocess
import sys
from datetime import datetime

import aiohttp
from rich.console import Console
from rich.table import Table

console = Console()


async def check_node(port):
    """Check node health via WebSocket health endpoint."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://localhost:{port + 11}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "port": port,
                        "status": "healthy",
                        "peers": data.get("peers", 0),
                    }
    except (aiohttp.ClientError, asyncio.TimeoutError, Exception):
        pass
    return {"port": port, "status": "unhealthy", "peers": 0}


async def health():
    """Run health check on all nodes."""
    console.print("🔍 Running Health Check...", style="bold blue")

    tasks = [check_node(9933), check_node(9934)]
    results = await asyncio.gather(*tasks)

    table = Table(title="Node Status")
    table.add_column("Node", style="cyan")
    table.add_column("Port", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Peers", style="yellow")

    for i, result in enumerate(results):
        status_color = "green" if result["status"] == "healthy" else "red"
        table.add_row(
            f"subtensor-{i}",
            str(result["port"]),
            f"[{status_color}]{result['status']}[/{status_color}]",
            str(result["peers"]),
        )

    console.print(table)


def status():
    """Show quick status."""
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "name=subtensor",
                "--format",
                "table {{.Names}}\t{{.Status}}",
            ],
            capture_output=True,
            text=True,
        )
        console.print("Docker Status:")
        console.print(result.stdout)
    except Exception as e:
        console.print(f"Error: {e}", style="red")


def logs():
    """Show recent logs."""
    try:
        subprocess.run(["docker", "logs", "--tail", "50", "subtensor-0"])
    except Exception as e:
        console.print(f"Error: {e}", style="red")


def restart():
    """Restart Subtensor services."""
    console.print("Restarting Subtensor services...", style="yellow")
    try:
        subprocess.run(["systemctl", "restart", "subtensor"], check=True)
        console.print("✅ Services restarted", style="green")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


def backup():
    """Create backup."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"subtensor_backup_{timestamp}"
    console.print(f"Creating backup: {backup_name}", style="yellow")

    try:
        subprocess.run(
            [
                "docker",
                "exec",
                "subtensor-0",
                "cp",
                "-r",
                "/tmp/blockchain",
                f"/tmp/{backup_name}",
            ],
            check=True,
        )
        console.print(f"✅ Backup created: {backup_name}", style="green")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


def snapshot():
    """Take a compressed snapshot."""
    import os

    date_str = datetime.now().strftime("%Y-%m-%d")
    snapshot_name = f"snapshot_{date_str}.tar.lz4"
    backup_dir = "/opt/subtensor/backups"
    chains_path = (
        "/var/snap/docker/common/var-lib-docker/volumes/node-0-volume/_data/chains"
    )

    console.print(f"Creating snapshot: {snapshot_name}", style="yellow")
    console.print(
        "⚠️  This will temporarily stop services for clean snapshot", style="yellow"
    )

    try:
        # Create backup directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)

        # Stop services for clean snapshot
        console.print("Stopping services...", style="yellow")
        subprocess.run(["systemctl", "stop", "subtensor"], check=True)

        # Create snapshot
        console.print("Creating compressed snapshot...", style="yellow")
        subprocess.run(
            [
                "tar",
                "-I",
                "lz4",
                "-cf",
                f"{backup_dir}/{snapshot_name}",
                chains_path,
                "--warning=no-file-changed",
            ],
            check=True,
        )

        # Restart services
        console.print("Restarting services...", style="yellow")
        subprocess.run(["systemctl", "start", "subtensor"], check=True)

        # Get file size
        size = os.path.getsize(f"{backup_dir}/{snapshot_name}")
        size_mb = size / (1024 * 1024)

        console.print(
            f"✅ Snapshot created: {backup_dir}/{snapshot_name} ({size_mb:.1f} MB)",
            style="green",
        )
    except Exception as e:
        # Make sure to restart services even if snapshot fails
        try:
            subprocess.run(["systemctl", "start", "subtensor"], check=True)
        except Exception:
            pass
        console.print(f"❌ Error: {e}", style="red")


def restore():
    """Restore from snapshot."""
    import os

    backup_dir = "/opt/subtensor/backups"
    restore_path = "/var/snap/docker/common/var-lib-docker/volumes/node-0-volume/_data"

    try:
        if not os.path.exists(backup_dir):
            console.print(f"❌ Backup directory not found: {backup_dir}", style="red")
            return

        snapshots = [f for f in os.listdir(backup_dir) if f.endswith(".tar.lz4")]
        if not snapshots:
            console.print("❌ No snapshots found", style="red")
            return

        console.print("Available snapshots:")
        for i, snapshot in enumerate(snapshots, 1):
            console.print(f"  {i}. {snapshot}")

        choice = input("\nSelect snapshot number (or press Enter to cancel): ")
        if not choice:
            console.print("Cancelled", style="yellow")
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(snapshots):
                snapshot_file = snapshots[idx]
                console.print(f"Restoring from: {snapshot_file}", style="yellow")
                console.print(
                    "⚠️  This will stop services and replace current data!", style="red"
                )
                confirm = input("Continue? (y/N): ")
                if confirm.lower() != "y":
                    console.print("Cancelled", style="yellow")
                    return

                # Stop services
                console.print("Stopping services...", style="yellow")
                subprocess.run(["systemctl", "stop", "subtensor"], check=True)

                # Remove existing chains data
                console.print("Removing existing data...", style="yellow")
                subprocess.run(["rm", "-rf", f"{restore_path}/chains"], check=True)

                # Extract snapshot
                console.print("Extracting snapshot...", style="yellow")
                subprocess.run(
                    [
                        "tar",
                        "-I",
                        "lz4",
                        "-xf",
                        f"{backup_dir}/{snapshot_file}",
                        "-C",
                        restore_path,
                        "--strip-components=6",
                    ],
                    check=True,
                )

                # Start services
                console.print("Starting services...", style="yellow")
                subprocess.run(["systemctl", "start", "subtensor"], check=True)

                console.print("✅ Restore completed", style="green")
            else:
                console.print("❌ Invalid selection", style="red")
        except ValueError:
            console.print("❌ Invalid selection", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


def main():
    if len(sys.argv) < 2:
        console.print("Usage: subtensor-admin <command>")
        console.print(
            "Commands: health, status, logs, restart, backup, snapshot, restore"
        )
        return

    command = sys.argv[1]

    if command == "health":
        asyncio.run(health())
    elif command == "status":
        status()
    elif command == "logs":
        logs()
    elif command == "restart":
        restart()
    elif command == "backup":
        backup()
    elif command == "snapshot":
        snapshot()
    elif command == "restore":
        restore()
    else:
        console.print(f"Unknown command: {command}", style="red")


if __name__ == "__main__":
    main()
