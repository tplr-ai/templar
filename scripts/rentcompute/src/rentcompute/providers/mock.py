"""
Mock provider implementation for rentcompute.

This provider is used for development and testing.
"""

import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from rentcompute.providers.base import BaseProvider, Machine, Pod


class MockProvider(BaseProvider):
    """Mock provider implementation for testing and development."""

    def __init__(self) -> None:
        """Initialize the mock provider."""
        self._authenticated = False
        self._running_pods: Dict[str, Pod] = {}

    @property
    def name(self) -> str:
        """Get provider name.

        Returns:
            Name of the provider
        """
        return "MockCloud"

    def authenticate(self, api_key: str) -> bool:
        """Authenticate with the mock provider.

        Args:
            api_key: API key for authentication

        Returns:
            True if authentication is successful
        """
        # Mock provider accepts any API key
        self._authenticated = True
        return True

    def search_machines(
        self,
        filters: Dict[str, Any],
        name_pattern: Optional[str] = None,
    ) -> List[Machine]:
        """Search for available machines with given filters.

        Args:
            filters: Dictionary of filter criteria (gpu, price, etc.)
            name_pattern: Optional pattern to filter by name

        Returns:
            List of matching Machine objects
        """
        if not self._authenticated:
            raise ValueError("Not authenticated. Call authenticate() first.")

        # Extract filters
        gpu_filters = filters.get("gpu", {})
        price_filters = filters.get("price", {})

        min_gpus = gpu_filters.get("min")
        max_gpus = gpu_filters.get("max")
        gpu_type = gpu_filters.get("type")
        min_price = price_filters.get("min")
        max_price = price_filters.get("max")

        # Mock data for different machine types
        all_machines = [
            Machine(
                id="m-a1b2c3d4",
                name="gpu-small",
                provider_name=self.name,
                gpu_type="v100",
                gpu_count=1,
                hourly_rate=0.99,
            ),
            Machine(
                id="m-e5f6g7h8",
                name="gpu-medium",
                provider_name=self.name,
                gpu_type="a100",
                gpu_count=2,
                hourly_rate=2.50,
            ),
            Machine(
                id="m-i9j0k1l2",
                name="gpu-large",
                provider_name=self.name,
                gpu_type="a100",
                gpu_count=4,
                hourly_rate=4.75,
            ),
            Machine(
                id="m-m3n4o5p6",
                name="gpu-xlarge",
                provider_name=self.name,
                gpu_type="h100",
                gpu_count=8,
                hourly_rate=10.50,
            ),
            Machine(
                id="m-q7r8s9t0",
                name="gpu-inference",
                provider_name=self.name,
                gpu_type="t4",
                gpu_count=4,
                hourly_rate=1.75,
            ),
            Machine(
                id="m-u1v2w3x4",
                name="gpu-training",
                provider_name=self.name,
                gpu_type="h100",
                gpu_count=2,
                hourly_rate=5.25,
            ),
        ]

        # Filter the machines based on the criteria
        results = []

        for machine in all_machines:
            # Apply filters
            if min_gpus is not None and machine.gpu_count < min_gpus:
                continue

            if max_gpus is not None and machine.gpu_count > max_gpus:
                continue

            if gpu_type is not None and machine.gpu_type.lower() != gpu_type.lower():
                continue

            if min_price is not None and machine.hourly_rate < min_price:
                continue

            if max_price is not None and machine.hourly_rate > max_price:
                continue

            if (
                name_pattern is not None
                and name_pattern.lower() not in machine.name.lower()
            ):
                continue

            # All filters passed, add to results
            results.append(machine)

        return results

    def start_machine(
        self,
        machine_id: str,
        name: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
    ) -> Optional[Pod]:
        """Start a machine and return a Pod object.

        Args:
            machine_id: ID of the machine to start
            name: Optional custom name for the machine
            ssh_key_path: Optional path to SSH public key file

        Returns:
            Pod object if successful, None otherwise
        """
        if not self._authenticated:
            raise ValueError("Not authenticated. Call authenticate() first.")

        # Find the machine by ID
        machines = self.search_machines({}, None)
        machine = next((m for m in machines if m.id == machine_id), None)

        if not machine:
            return None

        # Generate a unique ID for the pod
        pod_id = f"p-{uuid.uuid4().hex[:8]}"

        # Use provided name or generate one
        pod_name = (
            name if name else f"{machine.name}-{datetime.now().strftime('%m%d%H%M')}"
        )

        # Create a new pod
        pod = Pod(
            id=pod_id,
            name=pod_name,
            host=f"{pod_id}.mockcloud.example.com",
            user="ubuntu",
            port=22,
            key_path=ssh_key_path or "~/.ssh/id_rsa.pub",
            status="running",
            hourly_rate=machine.hourly_rate,
            gpu_type=machine.gpu_type,
            gpu_count=machine.gpu_count,
            provider_name=self.name,
        )

        # Store the pod
        self._running_pods[pod_id] = pod

        return pod

    def list_pods(self) -> List[Pod]:
        """List all running pods for this provider.

        Returns:
            List of Pod objects
        """
        if not self._authenticated:
            raise ValueError("Not authenticated. Call authenticate() first.")

        return list(self._running_pods.values())

    def stop_pod(self, pod_id: str) -> bool:
        """Stop a running pod.

        Args:
            pod_id: ID of the pod to stop

        Returns:
            True if stop was successful, False otherwise
        """
        if not self._authenticated:
            raise ValueError("Not authenticated. Call authenticate() first.")

        if pod_id in self._running_pods:
            # Update status to stopped
            self._running_pods[pod_id].status = "stopped"
            return True

        return False
