"""
Configuration management for rentcompute.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from rentcompute.providers.factory import ProviderFactory
from rentcompute.providers.base import BaseProvider


class Config:
    """Configuration manager for rentcompute."""

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        """Initialize configuration manager.

        Args:
            config_dir: Configuration directory (defaults to ~/.rentcompute)
        """
        if config_dir is None:
            self.config_dir = Path.home() / ".rentcompute"
        else:
            self.config_dir = config_dir

        self.credentials_file = self.config_dir / "credentials.yaml"
        self.instances_file = self.config_dir / "instances.yaml"
        self.config_file = self.config_dir / "config.yaml"

        # Create config directory if it doesn't exist
        self._ensure_config_dir()

        # Initialize provider
        self._provider: Optional[BaseProvider] = None

    def _ensure_config_dir(self) -> None:
        """Ensure the configuration directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def save_credentials(self, api_key: str, provider: str = "celium") -> None:
        """Save API credentials to credentials file.

        Args:
            api_key: API key for authentication
            provider: Name of the provider (defaults to celium)
        """
        credentials = {"api_key": api_key, "provider": provider}
        with open(self.credentials_file, "w") as f:
            yaml.dump(credentials, f)

        # Set secure permissions
        os.chmod(self.credentials_file, 0o600)

    def load_credentials(self) -> Dict[str, Any]:
        """Load API credentials from credentials file.

        Returns:
            Dict containing credentials

        Raises:
            FileNotFoundError: If credentials file doesn't exist
        """
        if not self.credentials_file.exists():
            raise FileNotFoundError(
                'Credentials file not found. Please run "rentcompute login"'
            )

        with open(self.credentials_file, "r") as f:
            return yaml.safe_load(f)

    def get_provider(self) -> BaseProvider:
        """Get the configured provider instance.

        Returns:
            Provider instance

        Raises:
            ValueError: If credentials are not set
        """
        if self._provider is not None:
            return self._provider

        # Load credentials and get provider
        credentials = self.load_credentials()
        provider_name = credentials.get("provider", "celium")
        api_key = credentials.get("api_key")

        if not api_key:
            raise ValueError("API key not found in credentials")

        # Get provider instance
        provider = ProviderFactory.get_provider(provider_name)

        # Authenticate with provider
        provider.authenticate(api_key)

        # Cache provider
        self._provider = provider

        return provider

    def save_instance(self, instance_id: str, instance_data: Dict[str, Any]) -> None:
        """Save instance data to instances file.

        Args:
            instance_id: Instance ID
            instance_data: Instance data including SSH connection details
        """
        instances = self.load_instances()
        instances[instance_id] = instance_data

        with open(self.instances_file, "w") as f:
            yaml.dump(instances, f)

    def load_instances(self) -> Dict[str, Any]:
        """Load instances data from instances file.

        Returns:
            Dict containing instances data
        """
        if not self.instances_file.exists():
            return {}

        with open(self.instances_file, "r") as f:
            instances = yaml.safe_load(f)

        return instances or {}

    def remove_instance(self, instance_id: str) -> None:
        """Remove instance from instances file.

        Args:
            instance_id: Instance ID to remove

        Raises:
            KeyError: If instance ID doesn't exist
        """
        instances = self.load_instances()
        if instance_id not in instances:
            raise KeyError(f"Instance {instance_id} not found")

        del instances[instance_id]
        with open(self.instances_file, "w") as f:
            yaml.dump(instances, f)
