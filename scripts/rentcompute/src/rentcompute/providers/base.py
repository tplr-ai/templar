"""
Base provider interface for rentcompute.

This module defines the BaseProvider abstract class that all provider
implementations must subclass.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class Machine:
    """Represents an available machine that can be started."""

    def __init__(
        self,
        id: str,
        name: str,
        provider_name: str,
        gpu_type: str,
        gpu_count: int,
        hourly_rate: float,
    ) -> None:
        """Initialize a Machine object.

        Args:
            id: Unique identifier for the machine
            name: Human-readable name for the machine
            provider_name: Name of the cloud provider
            gpu_type: Type of GPU (e.g., h100, a100)
            gpu_count: Number of GPUs
            hourly_rate: Cost per hour in USD
        """
        self.id = id
        self.name = name
        self.provider_name = provider_name
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        self.hourly_rate = hourly_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert machine to dictionary representation.

        Returns:
            Dictionary representation of the machine
        """
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider_name,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "hourly_rate": self.hourly_rate,
        }


class Pod:
    """Represents a running machine instance (pod)."""

    def __init__(
        self,
        id: str,
        name: str,
        host: str,
        user: str,
        port: int,
        key_path: str,
        status: str,
        hourly_rate: float,
        gpu_type: str,
        gpu_count: int,
        provider_name: str,
    ) -> None:
        """Initialize a Pod object.

        Args:
            id: Unique identifier for the pod
            name: Human-readable name for the pod
            host: Hostname or IP address
            user: SSH username
            port: SSH port
            key_path: Path to SSH key
            status: Current status (e.g., running, stopped)
            hourly_rate: Cost per hour in USD
            gpu_type: Type of GPU
            gpu_count: Number of GPUs
            provider_name: Name of the cloud provider
        """
        self.id = id
        self.name = name
        self.host = host
        self.user = user
        self.port = port
        self.key_path = key_path
        self.status = status
        self.hourly_rate = hourly_rate
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        self.provider_name = provider_name

    def to_dict(self) -> Dict[str, Any]:
        """Convert pod to dictionary representation.

        Returns:
            Dictionary representation of the pod
        """
        return {
            "id": self.id,
            "name": self.name,
            "host": self.host,
            "user": self.user,
            "port": self.port,
            "key_path": self.key_path,
            "status": self.status,
            "hourly_rate": self.hourly_rate,
            "gpu": {
                "type": self.gpu_type,
                "count": self.gpu_count,
            },
            "provider": self.provider_name,
        }


class BaseProvider(ABC):
    """Base abstract provider interface.

    All cloud provider implementations must inherit from this class
    and implement its abstract methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get provider name.

        Returns:
            String name of the provider
        """
        pass

    @abstractmethod
    def authenticate(self, api_key: str) -> bool:
        """Authenticate with the provider using API key.

        Args:
            api_key: API key for authentication

        Returns:
            True if authentication is successful, False otherwise
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def list_pods(self) -> List[Pod]:
        """List all running pods for this provider.

        Returns:
            List of Pod objects
        """
        pass

    @abstractmethod
    def stop_pod(self, pod_id: str) -> tuple[bool, str]:
        """Stop a running pod.

        Args:
            pod_id: ID of the pod to stop

        Returns:
            Tuple of (success, error_message)
            - success: True if stop was successful, False otherwise
            - error_message: Error message if failed, empty string if successful
        """
        pass
