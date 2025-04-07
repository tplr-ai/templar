"""
Login command implementation.
"""

import getpass
from typing import List

from rentcompute.config import Config
from rentcompute.providers.factory import ProviderFactory


def run(config: Config) -> None:
    """Run the login command.

    Args:
        config: Configuration manager
    """
    # Get available providers
    available_providers = ProviderFactory.get_available_providers()

    # Select provider
    provider = _select_provider(available_providers)

    print(f"Logging in to {provider}...")
    api_key = getpass.getpass("Please enter your API key:")

    # Save credentials
    config.save_credentials(api_key, provider)
    print(f"API key for {provider} saved successfully.")


def _select_provider(available_providers: List[str]) -> str:
    """Prompt user to select a provider.

    Args:
        available_providers: List of available provider names

    Returns:
        Selected provider name
    """
    if not available_providers:
        return "celium"  # Default if no providers are available

    if len(available_providers) == 1:
        return available_providers[0]  # Only one provider, use it

    # Multiple providers, let user choose
    print("Available providers:")
    for i, provider in enumerate(available_providers, 1):
        print(f"{i}. {provider}")

    while True:
        try:
            choice = input(
                "Select provider (1-{0}) [1]: ".format(len(available_providers))
            )
            if not choice.strip():
                # Default to first provider
                return available_providers[0]

            index = int(choice) - 1
            if 0 <= index < len(available_providers):
                return available_providers[index]
            else:
                print(
                    f"Invalid choice. Please enter a number between 1 and {len(available_providers)}."
                )
        except ValueError:
            print("Invalid input. Please enter a number.")
