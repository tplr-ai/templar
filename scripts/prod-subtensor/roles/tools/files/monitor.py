#!/usr/bin/env python3
import asyncio
import subprocess
import sys

import aiohttp
from rich.console import Console
from rich.table import Table

console = Console()


async def check_sync(port):
    """Check node sync status."""
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
                        "syncing": data.get("is_syncing", False),
                        "peers": data.get("peers", 0),
                        "best_block": data.get("best_block", "unknown"),
                    }
    except Exception:
        pass
    return {"port": port, "syncing": False, "peers": 0, "best_block": "unknown"}


async def sync():
    """Check sync status of all nodes."""
    console.print("🔄 Checking Sync Status...", style="bold blue")

    tasks = [check_sync(9933), check_sync(9934)]
    results = await asyncio.gather(*tasks)

    table = Table(title="Sync Status")
    table.add_column("Node", style="cyan")
    table.add_column("Port", style="magenta")
    table.add_column("Syncing", style="green")
    table.add_column("Peers", style="yellow")
    table.add_column("Best Block", style="blue")

    for i, result in enumerate(results):
        sync_color = "yellow" if result["syncing"] else "green"
        sync_status = "Yes" if result["syncing"] else "No"

        table.add_row(
            f"subtensor-{i}",
            str(result["port"]),
            f"[{sync_color}]{sync_status}[/{sync_color}]",
            str(result["peers"]),
            str(result["best_block"]),
        )

    console.print(table)


def metrics():
    """Show system metrics."""
    try:
        # CPU and memory
        result = subprocess.run(["top", "-bn1"], capture_output=True, text=True)
        lines = result.stdout.split("\n")[:10]
        console.print("System Metrics:")
        for line in lines:
            if "Cpu(s):" in line or "MiB Mem:" in line:
                console.print(line)

        # Disk space
        result = subprocess.run(["df", "-h", "/"], capture_output=True, text=True)
        console.print("\nDisk Usage:")
        console.print(result.stdout)

    except Exception as e:
        console.print(f"Error: {e}", style="red")


def dashboard():
    """Simple dashboard view."""
    console.print("📊 Subtensor Dashboard", style="bold green")
    console.print(
        "Use 'subtensor-admin health' and 'subtensor-monitor sync' for detailed info"
    )

    # Quick docker status
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "name=subtensor",
                "--format",
                "{{.Names}}: {{.Status}}",
            ],
            capture_output=True,
            text=True,
        )
        console.print("Container Status:")
        for line in result.stdout.strip().split("\n"):
            if line:
                console.print(f"  {line}")
    except Exception as e:
        console.print(f"Error: {e}", style="red")


def main():
    if len(sys.argv) < 2:
        console.print("Usage: subtensor-monitor <command>")
        console.print("Commands: sync, metrics, dashboard")
        return

    command = sys.argv[1]

    if command == "sync":
        asyncio.run(sync())
    elif command == "metrics":
        metrics()
    elif command == "dashboard":
        dashboard()
    else:
        console.print(f"Unknown command: {command}", style="red")


if __name__ == "__main__":
    main()
