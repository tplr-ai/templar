#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "bittensor>=7.0.0",
#     "requests>=2.28.0",
#     "pydantic>=2.0.0",
# ]
# ///
"""
Bittensor Subtensor Node Health Check Script

This script verifies that local subtensor nodes are properly synchronized
with the Bittensor mainnet by performing comprehensive health checks.

./scripts/prod-subtensor/health_check.py --verbose --local-endpoint=wss://0.1.2.3:9944 --test-subnets=1,3,18
"""

import argparse
import logging
import ssl
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import bittensor as bt


class HealthStatus(Enum):
    """Health check status enumeration."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    ERROR = "ERROR"


@dataclass
class HealthCheckResult:
    """Encapsulates the result of a health check operation."""

    name: str
    status: HealthStatus
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    error: Optional[Exception] = None


@dataclass
class NodeConfig:
    """Configuration for subtensor node endpoints and validation parameters.

    Implements immutable configuration state with explicit validation parameters
    for enterprise-grade health monitoring deployments.
    """

    local_endpoint: str
    mainnet_network: str = "finney"
    test_subnets: List[int] = field(default_factory=lambda: [3])
    sync_tolerance_blocks: int = 1
    test_address: str = "5HdTZQ6UXD7MWcRsMeExVwqAKKo4UwomUd662HvtXiZXkxmv"
    connection_timeout: int = 30
    verify_ssl: bool = True


class HealthChecker(ABC):
    """Abstract base class defining the health check contract.

    Implements the Strategy pattern to enable polymorphic health validation
    behaviors while maintaining consistent interface contracts.
    """

    def __init__(self, config: NodeConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.elapsed_ms = 0.0

    @abstractmethod
    def check_health(self) -> HealthCheckResult:
        """Execute the health check and return results.

        Returns:
            HealthCheckResult: Encapsulated validation outcome with metrics
        """
        pass

    @contextmanager
    def measure_time(self):
        """Context manager for high-precision execution time measurement.

        Uses monotonic clock to prevent issues with system time adjustments.
        """
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        self.elapsed_ms = (end_time - start_time) * 1000


class ConnectivityChecker(HealthChecker):
    """Verifies network connectivity and RPC endpoint availability.

    Validates that both local and mainnet subtensor endpoints are accessible
    and capable of serving blockchain state queries.
    """

    def check_health(self) -> HealthCheckResult:
        """Execute connectivity validation against configured endpoints.

        Returns:
            HealthCheckResult: Connection status with endpoint metrics
        """
        with self.measure_time():
            try:
                if (
                    not self.config.verify_ssl
                    and "wss://" in self.config.local_endpoint
                ):
                    pass

                local_subtensor = None
                mainnet_subtensor = None

                try:
                    local_subtensor = bt.subtensor(network=self.config.local_endpoint)
                except Exception as local_error:
                    if (
                        "CERTIFICATE_VERIFY_FAILED" in str(local_error)
                        and not self.config.verify_ssl
                    ):
                        try:
                            local_subtensor = bt.subtensor(
                                network=self.config.local_endpoint
                            )
                        except Exception:
                            raise local_error
                    else:
                        raise local_error

                try:
                    mainnet_subtensor = bt.subtensor(
                        network=self.config.mainnet_network
                    )
                except Exception as mainnet_error:
                    if local_subtensor:
                        local_subtensor.close()
                    raise mainnet_error

                local_block = local_subtensor.get_current_block()
                mainnet_block = mainnet_subtensor.get_current_block()

                metrics = {
                    "local_block": local_block,
                    "mainnet_block": mainnet_block,
                    "local_endpoint": self.config.local_endpoint,
                    "mainnet_network": self.config.mainnet_network,
                }

                local_subtensor.close()
                mainnet_subtensor.close()

                return HealthCheckResult(
                    name="connectivity",
                    status=HealthStatus.HEALTHY,
                    message="Successfully connected to both local and mainnet nodes",
                    metrics=metrics,
                    elapsed_ms=self.elapsed_ms,
                )

            except ssl.SSLError as e:
                ssl_message = str(e)
                if "CERTIFICATE_VERIFY_FAILED" in ssl_message:
                    suggestion = (
                        " (Try using --no-verify-ssl flag for self-signed certificates)"
                    )
                else:
                    suggestion = ""

                return HealthCheckResult(
                    name="connectivity",
                    status=HealthStatus.ERROR,
                    message=f"SSL connection failed: {ssl_message}{suggestion}",
                    error=e,
                    elapsed_ms=self.elapsed_ms,
                )
            except Exception as e:
                return HealthCheckResult(
                    name="connectivity",
                    status=HealthStatus.ERROR,
                    message=f"Connection failed: {str(e)}",
                    error=e,
                    elapsed_ms=self.elapsed_ms,
                )


class SynchronizationChecker(HealthChecker):
    """Validates blockchain synchronization state between local and mainnet nodes.

    Implements strict synchronization verification with configurable tolerance
    to account for network propagation delays and consensus finality.
    """

    def check_health(self) -> HealthCheckResult:
        """Perform block height comparison for synchronization validation.

        Returns:
            HealthCheckResult: Synchronization status with block metrics
        """
        with self.measure_time():
            try:
                local_subtensor = bt.subtensor(network=self.config.local_endpoint)
                mainnet_subtensor = bt.subtensor(network=self.config.mainnet_network)

                local_block = local_subtensor.get_current_block()
                mainnet_block = mainnet_subtensor.get_current_block()

                block_diff = abs(mainnet_block - local_block)
                is_synced = block_diff <= self.config.sync_tolerance_blocks

                metrics = {
                    "local_block": local_block,
                    "mainnet_block": mainnet_block,
                    "block_difference": block_diff,
                    "sync_tolerance": self.config.sync_tolerance_blocks,
                    "is_synced": is_synced,
                }

                local_subtensor.close()
                mainnet_subtensor.close()

                if is_synced:
                    status = HealthStatus.HEALTHY
                    message = f"Nodes synchronized (diff: {block_diff} blocks)"
                elif block_diff <= self.config.sync_tolerance_blocks * 2:
                    status = HealthStatus.DEGRADED
                    message = f"Nodes slightly out of sync (diff: {block_diff} blocks)"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = (
                        f"Nodes significantly out of sync (diff: {block_diff} blocks)"
                    )

                return HealthCheckResult(
                    name="synchronization",
                    status=status,
                    message=message,
                    metrics=metrics,
                    elapsed_ms=self.elapsed_ms,
                )

            except ssl.SSLError as e:
                ssl_message = str(e)
                if "CERTIFICATE_VERIFY_FAILED" in ssl_message:
                    suggestion = (
                        " (Try using --no-verify-ssl flag for self-signed certificates)"
                    )
                else:
                    suggestion = ""

                return HealthCheckResult(
                    name="synchronization",
                    status=HealthStatus.ERROR,
                    message=f"SSL connection failed: {ssl_message}{suggestion}",
                    error=e,
                    elapsed_ms=self.elapsed_ms,
                )
            except Exception as e:
                return HealthCheckResult(
                    name="synchronization",
                    status=HealthStatus.ERROR,
                    message=f"Synchronization check failed: {str(e)}",
                    error=e,
                    elapsed_ms=self.elapsed_ms,
                )


class MetagraphChecker(HealthChecker):
    """Validates metagraph operations and subnet state consistency.

    Performs lightweight metagraph queries to verify subnet accessibility
    and data consistency between local and mainnet endpoints.
    """

    def check_health(self) -> HealthCheckResult:
        """Execute metagraph validation across configured test subnets.

        Returns:
            HealthCheckResult: Metagraph validation status with subnet metrics
        """
        with self.measure_time():
            try:
                results = {}
                local_subtensor = bt.subtensor(network=self.config.local_endpoint)
                mainnet_subtensor = bt.subtensor(network=self.config.mainnet_network)

                for netuid in self.config.test_subnets:
                    try:
                        local_meta = local_subtensor.metagraph(netuid=netuid, lite=True)
                        mainnet_meta = mainnet_subtensor.metagraph(
                            netuid=netuid, lite=True
                        )

                        local_neurons = (
                            len(local_meta.uids) if hasattr(local_meta, "uids") else 0
                        )
                        mainnet_neurons = (
                            len(mainnet_meta.uids)
                            if hasattr(mainnet_meta, "uids")
                            else 0
                        )

                        results[f"subnet_{netuid}"] = {
                            "local_neurons": local_neurons,
                            "mainnet_neurons": mainnet_neurons,
                            "neurons_match": local_neurons == mainnet_neurons,
                            "local_block": getattr(local_meta, "block", None),
                            "mainnet_block": getattr(mainnet_meta, "block", None),
                        }

                    except Exception as subnet_error:
                        results[f"subnet_{netuid}"] = {
                            "error": str(subnet_error),
                            "status": "failed",
                        }

                local_subtensor.close()
                mainnet_subtensor.close()

                successful_subnets = sum(
                    1
                    for r in results.values()
                    if isinstance(r, dict) and "error" not in r
                )
                total_subnets = len(self.config.test_subnets)

                if successful_subnets == total_subnets:
                    status = HealthStatus.HEALTHY
                    message = f"Metagraph operations successful on all {total_subnets} subnets"
                elif successful_subnets > 0:
                    status = HealthStatus.DEGRADED
                    message = f"Metagraph operations successful on {successful_subnets}/{total_subnets} subnets"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = "Metagraph operations failed on all subnets"

                return HealthCheckResult(
                    name="metagraph",
                    status=status,
                    message=message,
                    metrics=results,
                    elapsed_ms=self.elapsed_ms,
                )

            except ssl.SSLError as e:
                ssl_message = str(e)
                if "CERTIFICATE_VERIFY_FAILED" in ssl_message:
                    suggestion = (
                        " (Try using --no-verify-ssl flag for self-signed certificates)"
                    )
                else:
                    suggestion = ""

                return HealthCheckResult(
                    name="metagraph",
                    status=HealthStatus.ERROR,
                    message=f"SSL connection failed: {ssl_message}{suggestion}",
                    error=e,
                    elapsed_ms=self.elapsed_ms,
                )
            except Exception as e:
                return HealthCheckResult(
                    name="metagraph",
                    status=HealthStatus.ERROR,
                    message=f"Metagraph check failed: {str(e)}",
                    error=e,
                    elapsed_ms=self.elapsed_ms,
                )


class WalletChecker(HealthChecker):
    """Validates wallet operations and balance query functionality.

    Tests basic wallet operations without requiring actual key material
    by performing balance queries against a configured test address.
    """

    def check_health(self) -> HealthCheckResult:
        """Execute wallet operation validation using balance queries.

        Returns:
            HealthCheckResult: Wallet operation status with query metrics
        """
        with self.measure_time():
            try:
                local_subtensor = bt.subtensor(network=self.config.local_endpoint)
                mainnet_subtensor = bt.subtensor(network=self.config.mainnet_network)

                local_balance_success = False
                mainnet_balance_success = False

                try:
                    local_subtensor.get_balance(self.config.test_address)
                    local_balance_success = True
                except Exception:
                    pass

                try:
                    mainnet_subtensor.get_balance(self.config.test_address)
                    mainnet_balance_success = True
                except Exception:
                    pass

                metrics = {
                    "local_balance_query": local_balance_success,
                    "mainnet_balance_query": mainnet_balance_success,
                    "test_address": self.config.test_address,
                }

                local_subtensor.close()
                mainnet_subtensor.close()

                if local_balance_success and mainnet_balance_success:
                    status = HealthStatus.HEALTHY
                    message = "Wallet operations successful on both endpoints"
                elif local_balance_success or mainnet_balance_success:
                    status = HealthStatus.DEGRADED
                    message = "Wallet operations partially successful"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = "Wallet operations failed on both endpoints"

                return HealthCheckResult(
                    name="wallet",
                    status=status,
                    message=message,
                    metrics=metrics,
                    elapsed_ms=self.elapsed_ms,
                )

            except ssl.SSLError as e:
                ssl_message = str(e)
                if "CERTIFICATE_VERIFY_FAILED" in ssl_message:
                    suggestion = (
                        " (Try using --no-verify-ssl flag for self-signed certificates)"
                    )
                else:
                    suggestion = ""

                return HealthCheckResult(
                    name="wallet",
                    status=HealthStatus.ERROR,
                    message=f"SSL connection failed: {ssl_message}{suggestion}",
                    error=e,
                    elapsed_ms=self.elapsed_ms,
                )
            except Exception as e:
                return HealthCheckResult(
                    name="wallet",
                    status=HealthStatus.ERROR,
                    message=f"Wallet check failed: {str(e)}",
                    error=e,
                    elapsed_ms=self.elapsed_ms,
                )


class SubtensorHealthMonitor:
    """Orchestrates comprehensive health validation for subtensor node infrastructure.

    Implements the Facade pattern to provide simplified interface for complex
    health monitoring operations while maintaining separation of concerns.
    """

    def __init__(self, config: NodeConfig):
        self.config = config
        self.checkers: List[HealthChecker] = [
            ConnectivityChecker(config),
            SynchronizationChecker(config),
            MetagraphChecker(config),
            WalletChecker(config),
        ]
        self.setup_logging()

    def setup_logging(self) -> None:
        """Configure structured logging for health monitoring operations."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def run_all_checks(self) -> List[HealthCheckResult]:
        """Execute comprehensive health validation suite.

        Returns:
            List[HealthCheckResult]: Complete validation results
        """
        results = []

        print("Starting Bittensor Subtensor Node Health Checks")
        print("=" * 60)

        for checker in self.checkers:
            print(f"\nRunning {checker.__class__.__name__}...")
            result = checker.check_health()
            results.append(result)

            status_indicators = {
                HealthStatus.HEALTHY: "[PASS]",
                HealthStatus.DEGRADED: "[WARN]",
                HealthStatus.UNHEALTHY: "[FAIL]",
                HealthStatus.ERROR: "[ERROR]",
            }

            print(
                f"{status_indicators[result.status]} {result.name.title()}: {result.message}"
            )
            print(f"   Completed in {result.elapsed_ms:.2f}ms")

        return results

    def generate_report(self, results: List[HealthCheckResult]) -> None:
        """Generate comprehensive operational health report.

        Args:
            results: Complete health check validation results
        """
        print("\n" + "=" * 60)
        print("HEALTH CHECK SUMMARY")
        print("=" * 60)

        overall_status = self.calculate_overall_status(results)
        status_symbols = {
            HealthStatus.HEALTHY: "[OK]",
            HealthStatus.DEGRADED: "[DEGRADED]",
            HealthStatus.UNHEALTHY: "[CRITICAL]",
            HealthStatus.ERROR: "[ERROR]",
        }

        print(
            f"\nOverall Status: {status_symbols[overall_status]} {overall_status.value}"
        )
        print(f"Local Endpoint: {self.config.local_endpoint}")
        print(f"Mainnet Network: {self.config.mainnet_network}")

        print("\nDetailed Results:")
        for result in results:
            print(f"\n{result.name.upper()}")
            print(f"   Status: {result.status.value}")
            print(f"   Message: {result.message}")
            print(f"   Duration: {result.elapsed_ms:.2f}ms")

            if result.metrics:
                print("   Metrics:")
                for key, value in result.metrics.items():
                    print(f"      {key}: {value}")

            if result.error:
                print(f"   Error: {result.error}")

        self.generate_recommendations(results)

    def calculate_overall_status(
        self, results: List[HealthCheckResult]
    ) -> HealthStatus:
        """Calculate aggregate health status using worst-case analysis.

        Args:
            results: Individual health check results

        Returns:
            HealthStatus: Overall system health status
        """
        if any(r.status == HealthStatus.ERROR for r in results):
            return HealthStatus.ERROR
        elif any(r.status == HealthStatus.UNHEALTHY for r in results):
            return HealthStatus.UNHEALTHY
        elif any(r.status == HealthStatus.DEGRADED for r in results):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def generate_recommendations(self, results: List[HealthCheckResult]) -> None:
        """Generate operational recommendations based on validation results.

        Args:
            results: Complete health check validation results
        """
        print("\nRECOMMENDATIONS")
        print("-" * 30)

        connectivity_result = next(
            (r for r in results if r.name == "connectivity"), None
        )
        sync_result = next((r for r in results if r.name == "synchronization"), None)

        if connectivity_result and connectivity_result.status == HealthStatus.ERROR:
            print("- Verify local subtensor node process status and accessibility")
            print("- Check network connectivity and firewall configuration")
            print("- Validate RPC endpoint configuration and port availability")

        if sync_result and sync_result.status != HealthStatus.HEALTHY:
            block_diff = sync_result.metrics.get("block_difference", 0)
            if block_diff > self.config.sync_tolerance_blocks:
                print(
                    f"- Local node is {block_diff} blocks behind - initiate resynchronization"
                )
                print("- Monitor system resources (CPU, memory, disk I/O)")
                print("- Consider adjusting sync strategy or peer connections")

        healthy_count = sum(1 for r in results if r.status == HealthStatus.HEALTHY)
        total_count = len(results)

        if healthy_count == total_count:
            print("All systems operational - subtensor nodes are healthy")
        else:
            print(
                f"{total_count - healthy_count}/{total_count} checks require attention"
            )


def parse_subnet_list(subnet_str: str) -> List[int]:
    """Parse comma-separated subnet list with validation.

    Args:
        subnet_str: Comma-separated subnet identifiers (e.g., "1,3,18")

    Returns:
        List[int]: Validated subnet identifiers

    Raises:
        argparse.ArgumentTypeError: If subnet identifiers are invalid
    """
    try:
        subnets = [int(s.strip()) for s in subnet_str.split(",") if s.strip()]
        if not subnets:
            raise argparse.ArgumentTypeError("At least one subnet must be specified")

        for subnet_id in subnets:
            if subnet_id < 0 or subnet_id > 255:
                raise argparse.ArgumentTypeError(f"Invalid subnet ID: {subnet_id}")

        return subnets
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid subnet list format: {subnet_str}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure command line argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Bittensor Subtensor Node Health Check Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --local-endpoint ws://localhost:9944
  %(prog)s --local-endpoint ws://10.0.1.100:9944 --test-address 5HdTZQ6UXD7MWcRsMeExVwqAKKo4UwomUd662HvtXiZXkxmv
  %(prog)s --local-endpoint wss://my-node.example.com:9944 --min-peer-count 100
  %(prog)s --local-endpoint ws://localhost:9944 --test-subnets 1,3,18 --sync-tolerance-blocks 2
  %(prog)s --local-endpoint ws://localhost:9944 --test-subnets 5 --sync-tolerance-blocks 0
        """,
    )

    parser.add_argument(
        "--local-endpoint",
        type=str,
        required=True,
        help="WebSocket endpoint URL for local subtensor node (e.g., ws://localhost:9944)",
    )

    parser.add_argument(
        "--mainnet-network",
        type=str,
        default="finney",
        help="Mainnet network identifier (default: finney)",
    )

    parser.add_argument(
        "--test-address",
        type=str,
        default="5HdTZQ6UXD7MWcRsMeExVwqAKKo4UwomUd662HvtXiZXkxmv",
        help="Test address for balance queries (default: Alice's test address)",
    )

    parser.add_argument(
        "--sync-tolerance-blocks",
        type=int,
        default=1,
        help="Maximum acceptable block difference for synchronization validation (default: 1)",
    )

    parser.add_argument(
        "--test-subnets",
        type=parse_subnet_list,
        default="3",
        help="Comma-separated list of subnet IDs to test (default: 3, example: 1,3,18)",
    )

    parser.add_argument(
        "--connection-timeout",
        type=int,
        default=30,
        help="Connection timeout in seconds (default: 30)",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging output"
    )

    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification for self-signed certificates",
    )

    return parser


def main():
    """Main entry point for health check script execution."""
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = NodeConfig(
        local_endpoint=args.local_endpoint,
        mainnet_network=args.mainnet_network,
        test_subnets=args.test_subnets,
        sync_tolerance_blocks=args.sync_tolerance_blocks,
        test_address=args.test_address,
        connection_timeout=args.connection_timeout,
        verify_ssl=not args.no_verify_ssl,
    )

    monitor = SubtensorHealthMonitor(config)
    results = monitor.run_all_checks()
    monitor.generate_report(results)

    overall_status = monitor.calculate_overall_status(results)
    exit_code = 0 if overall_status == HealthStatus.HEALTHY else 1
    exit(exit_code)


if __name__ == "__main__":
    main()
