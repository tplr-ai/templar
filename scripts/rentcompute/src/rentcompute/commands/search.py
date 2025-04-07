"""
Search command implementation.
"""

from typing import Any, Dict

from rentcompute.config import Config


def run(config: Config, instance_config: Dict[str, Any], name: str = None) -> None:
    """Run the search command to find available instances without starting them.

    Args:
        config: Configuration manager
        instance_config: Configuration including GPU and price requirements
        name: Optional name pattern to filter results
    """
    # Get the provider from config
    provider = config.get_provider()

    # Extract configuration values for display
    gpu_config = instance_config["gpu"]
    price_config = instance_config["price"]

    min_gpus = gpu_config["min"]
    max_gpus = gpu_config["max"]
    gpu_type = gpu_config["type"]
    min_price = price_config["min"]
    max_price = price_config["max"]

    # Check if all filter values are None
    gpu_filters = [min_gpus, max_gpus, gpu_type]
    price_filters = [min_price, max_price]
    all_filters = gpu_filters + price_filters

    # Build search criteria description for user feedback
    if all(v is None for v in all_filters) and name is None:
        print(f"Searching for all available compute instances from {provider.name}...")
    else:
        filter_desc = []

        if name is not None:
            filter_desc.append(f"Name: {name}")

        # GPU description
        gpu_desc = []
        if min_gpus is not None and max_gpus is not None:
            if min_gpus == max_gpus:
                gpu_desc.append(f"{min_gpus} GPUs")
            else:
                gpu_desc.append(f"{min_gpus}-{max_gpus} GPUs")
        elif min_gpus is not None:
            gpu_desc.append(f"at least {min_gpus} GPUs")
        elif max_gpus is not None:
            gpu_desc.append(f"up to {max_gpus} GPUs")

        if gpu_type is not None:
            gpu_desc.append(f"type {gpu_type}")

        if gpu_desc:
            filter_desc.append(f"GPU: {' '.join(gpu_desc)}")

        # Price description
        price_desc = []
        if min_price is not None and max_price is not None:
            if min_price == max_price:
                price_desc.append(f"${min_price}/hr")
            else:
                price_desc.append(f"${min_price}-${max_price}/hr")
        elif min_price is not None:
            price_desc.append(f"at least ${min_price}/hr")
        elif max_price is not None:
            price_desc.append(f"up to ${max_price}/hr")

        if price_desc:
            filter_desc.append(f"Price: {' '.join(price_desc)}")

        print(
            f"Searching for available instances from {provider.name} matching: {', '.join(filter_desc)}..."
        )

    # Search for machines using the provider
    search_results = provider.search_machines(instance_config, name)

    # Display results
    if not search_results:
        print("No matching instances found.")
        return

    # Print the results in a table
    print(f"\nFound {len(search_results)} matching instances:")

    # Define column widths with more space for longer names and GPU types
    name_width = 25
    id_width = 36
    provider_width = 11
    gpu_type_width = 25
    gpu_count_width = 9
    price_width = 10

    # Print table header
    print(
        f"{'Name':<{name_width}} | {'ID':<{id_width}} | {'Provider':<{provider_width}} | "
        f"{'GPU Type':<{gpu_type_width}} | {'GPU Count':<{gpu_count_width}} | {'Price ($/hr)':<{price_width}}"
    )

    # Calculate total width for divider
    total_width = (
        name_width
        + id_width
        + provider_width
        + gpu_type_width
        + gpu_count_width
        + price_width
        + 13
    )  # 13 for separators
    print("-" * total_width)

    for machine in search_results:
        machine_dict = machine.to_dict()

        # Truncate long values with ellipsis if needed
        name = machine_dict["name"]
        if len(name) > name_width:
            name = name[: name_width - 3] + "..."

        gpu_type = machine_dict["gpu_type"]
        if len(gpu_type) > gpu_type_width:
            gpu_type = gpu_type[: gpu_type_width - 3] + "..."

        print(
            f"{name:<{name_width}} | {machine_dict['id']:<{id_width}} | {machine_dict['provider']:<{provider_width}} | "
            f"{gpu_type:<{gpu_type_width}} | {machine_dict['gpu_count']:<{gpu_count_width}} | "
            f"${machine_dict['hourly_rate']:<{price_width - 1}.2f}"
        )

    print(
        "\nTo start an instance, use the start command with the same filters or specify exactly which instance:"
    )
    print("  rentcompute start <same-filters-as-search>")
