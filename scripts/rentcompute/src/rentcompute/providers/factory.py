"""
Provider factory for rentcompute.

This module contains the ProviderFactory class that manages provider
instantiation and retrieval.
"""

from typing import Dict, Type, List

from rentcompute.providers.base import BaseProvider
from rentcompute.providers.mock import MockProvider
from rentcompute.providers.celium import CeliumProvider


class ProviderFactory:
    """Factory for creating and retrieving provider instances."""

    _providers: Dict[str, Type[BaseProvider]] = {}
    _instances: Dict[str, BaseProvider] = {}

    @classmethod
    def register_provider(cls, provider_class: Type[BaseProvider]) -> None:
        """Register a provider class with the factory.

        Args:
            provider_class: Provider class to register
        """
        provider_name = provider_class().name.lower()
        cls._providers[provider_name] = provider_class

    @classmethod
    def get_provider(cls, name: str) -> BaseProvider:
        """Get or create a provider instance by name.

        Args:
            name: Name of the provider (case-insensitive)

        Returns:
            Provider instance

        Raises:
            ValueError: If provider is not registered
        """
        name_lower = name.lower()

        # Check if we already have an instance
        if name_lower in cls._instances:
            return cls._instances[name_lower]

        # Create a new instance if we have the provider registered
        if name_lower in cls._providers:
            instance = cls._providers[name_lower]()
            cls._instances[name_lower] = instance
            return instance

        raise ValueError(f"Provider '{name}' is not registered")

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get names of all registered providers.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())


# Register the available providers
ProviderFactory.register_provider(MockProvider)
ProviderFactory.register_provider(CeliumProvider)
