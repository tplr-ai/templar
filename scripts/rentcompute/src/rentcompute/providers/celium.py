"""
Celium provider implementation for rentcompute.

This provider interacts with the Celium Compute API for renting compute instances.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from rentcompute.providers.base import BaseProvider, Machine, Pod


class CeliumProvider(BaseProvider):
    """Provider implementation for Celium Compute."""

    API_BASE_URL = "https://celiumcompute.ai/api"

    def __init__(self) -> None:
        """Initialize the Celium provider."""
        self._api_key: Optional[str] = None
        self._authenticated = False
        self._logger = logging.getLogger(__name__)
        self._name_filter = None

    @property
    def name(self) -> str:
        """Get provider name.

        Returns:
            Name of the provider
        """
        return "Celium"

    def authenticate(self, api_key: str) -> bool:
        """Authenticate with Celium using API key.

        Args:
            api_key: API key for authentication

        Returns:
            True if authentication is successful, False otherwise
        """
        self._api_key = api_key

        # Validate the API key by making a test request
        try:
            # Make a small request to check if API key is valid
            response = self._make_request(
                "GET", "/executors", params={"page": 0, "size": 1}
            )
            self._authenticated = response.status_code == 200
            return self._authenticated
        except Exception as e:
            self._logger.error(f"Authentication failed: {e}")
            return False

    def search_machines(
        self,
        filters: dict[str, Any],
        name_pattern: Optional[str] = None,
    ) -> List[Machine]:
        """Search for available machines with given filters.

        Args:
            filters: Dictionary of filter criteria (gpu, price, etc.)
            name_pattern: Optional pattern to filter by name

        Returns:
            List of matching Machine objects
        """
        if not self._authenticated or not self._api_key:
            raise ValueError("Not authenticated. Call authenticate() first.")

        # Reset name filter before each search
        self._name_filter = None

        # Extract and prepare filters for API request
        params = self._prepare_search_params(filters, name_pattern)
        self._logger.debug(f"Search parameters: {params}")

        try:
            # Make request to Celium API
            self._logger.debug("Making request to /executors endpoint")
            response = self._make_request("GET", "/executors", params=params)
            self._logger.debug(f"Response status: {response.status_code}")

            if response.status_code != 200:
                self._logger.error(
                    f"API request failed with status {response.status_code}: {response.text}"
                )
                return []

            # Convert API response to Machine objects
            response_json = response.json()
            self._logger.debug(f"Response data type: {type(response_json)}")
            self._logger.debug(f"Response data preview: {str(response_json)[:200]}...")

            machines = self._convert_response_to_machines(response_json)
            self._logger.debug(f"Found {len(machines)} machines in total")
            return machines

        except Exception as e:
            self._logger.error(f"Error searching machines: {e}")
            return []

    def _fetch_pod_details(self, pod_id: str) -> dict[str, Any]:
        """Fetch details of a pod from the API.

        Args:
            pod_id: ID of the pod to fetch

        Returns:
            Dictionary containing pod details or empty dict if failed
        """
        try:
            pod_endpoint = f"/pods/{pod_id}"
            pod_response = self._make_request("GET", pod_endpoint)

            if pod_response.status_code == 200:
                pod_details = pod_response.json()
                self._logger.debug(f"Pod details response: {pod_details}")
                return pod_details
            else:
                self._logger.error(
                    f"Failed to fetch pod details: {pod_response.status_code} - {pod_response.text}"
                )
                return {}
        except Exception as e:
            self._logger.error(f"Error fetching pod details: {e}")
            return {}

    def _get_machine_details(self, machine_id: str) -> Optional[Machine]:
        """Get machine details by ID from search results.

        Args:
            machine_id: ID of the machine to find

        Returns:
            Machine object if found, None otherwise
        """
        try:
            search_results = self.search_machines({})
            for machine in search_results:
                if machine.id == machine_id:
                    return machine
            return None
        except Exception as e:
            self._logger.error(f"Error getting machine details: {e}")
            return None

    def _read_ssh_keys(self, ssh_key_path: Optional[str] = None) -> List[str]:
        """Read SSH public keys from provided path or default location.

        Args:
            ssh_key_path: Optional path to SSH public key file

        Returns:
            List of SSH public key strings
        """
        user_public_keys = []

        # If SSH key path is provided, read from that file
        if ssh_key_path:
            try:
                with open(ssh_key_path, "r") as f:
                    key_content = f.read().strip()
                    if key_content:
                        user_public_keys.append(key_content)
                        return user_public_keys
            except Exception as e:
                self._logger.error(f"Error reading SSH key from {ssh_key_path}: {e}")
                print(f"Error reading SSH key: {e}")

        # If no key provided or reading failed, try default location
        default_key_path = f"{str(Path.home())}/.ssh/id_rsa.pub"
        try:
            with open(default_key_path, "r") as f:
                key_content = f.read().strip()
                if key_content:
                    user_public_keys.append(key_content)
        except Exception as e:
            self._logger.error(
                f"Error reading default SSH key from {default_key_path}: {e}"
            )
            print("Error reading SSH key. Please provide a valid SSH public key file.")

        return user_public_keys

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
            ssh_key_path: Path to SSH public key file

        Returns:
            Pod object if successful, None otherwise
        """
        if not self._authenticated or not self._api_key:
            raise ValueError("Not authenticated. Call authenticate() first.")

        # Use the provided name or generate one using machine ID (last part)
        pod_name = name or f"pod-{machine_id.split('-')[-1]}"

        # Read SSH public keys
        user_public_keys = self._read_ssh_keys(ssh_key_path)
        if not user_public_keys:
            print("No SSH public key available. Please provide a valid SSH key.")
            return None

        # Prepare the request payload
        payload = {
            "pod_name": pod_name,
            # "template_id": "1742e309-5dcb-4de6-84f3-ad7190df4689",  # Hardcoded template ID
            "template_id": "9f1207a8-4633-412b-aab9-1256b9f94cad",  # Hardcoded template ID
            "user_public_key": user_public_keys,
        }

        try:
            # Make the POST request to start the machine
            endpoint = f"/executors/{machine_id}/rent"
            self._logger.debug(f"Making POST request to {endpoint}")
            response = self._make_request("POST", endpoint, json_data=payload)

            if response.status_code != 200:
                self._logger.error(
                    f"Failed to start machine: {response.status_code} - {response.text}"
                )
                print(f"Failed to start machine: {response.text}")
                return None

            # Parse the initial response
            response_data = response.json()
            pod_id = response_data.get("pod_id", "") or response_data.get(
                "id", machine_id
            )

            # Wait for pod to initialize and get connection details
            print("Waiting for pod to initialize...")
            # Initialize variables for continuous polling
            import datetime
            import time

            ssh_port = 22  # Default SSH port if port mapping not found
            ssh_host = None
            pod_ready = False
            start_time = datetime.datetime.now()
            timeout_minutes = 10  # Maximum wait time in minutes

            # Poll until we have connection details or hit timeout
            while not pod_ready:
                # Check if we've exceeded the timeout
                elapsed = datetime.datetime.now() - start_time
                if elapsed.total_seconds() > timeout_minutes * 60:
                    print(
                        f"Warning: Timed out after {timeout_minutes} minutes waiting for pod to initialize."
                    )
                    break

                time.sleep(10)  # Wait 10 seconds between attempts

                # Show status message
                elapsed_minutes = elapsed.total_seconds() / 60
                print(
                    f"Checking pod status... (elapsed: {elapsed_minutes:.1f} minutes)"
                )

                # Fetch pod details
                pod_details = self._fetch_pod_details(pod_id)

                if pod_details:
                    # Check for IP address and port mappings
                    executor_info = pod_details.get("executor", {})
                    ssh_host = executor_info.get("executor_ip_address")
                    ports_mapping = pod_details.get("ports_mapping", {})

                    # Check if we have both the IP address and SSH port mapping
                    if ssh_host and ports_mapping:
                        # Port mappings might be stored as string keys
                        for key, value in ports_mapping.items():
                            if str(key) == "22":
                                ssh_port = value
                                print(
                                    f"Pod is ready with SSH access at {ssh_host}:{ssh_port}"
                                )
                                pod_ready = True
                                break

                        # If we found a port mapping and host, mark as ready
                        if pod_ready:
                            break

            # Set default connection details if we haven't already
            ssh_user = "root"  # Default SSH user for containers
            gpu_type = None
            gpu_count = None
            hourly_rate = None

            if pod_details:
                # Extract GPU details (we already have ssh_host and ssh_port from our polling loop)

                # Extract GPU details
                gpu_type = pod_details.get("gpu_name")
                gpu_count_str = pod_details.get("gpu_count")
                if gpu_count_str and gpu_count_str.isdigit():
                    gpu_count = int(gpu_count_str)

                # Get price information
                executor_info = pod_details.get("executor", {})
                hourly_rate = executor_info.get("price_per_hour")

            # If we need additional details, get them from the machine
            if not ssh_host or not gpu_type or gpu_count is None or hourly_rate is None:
                machine_details = self._get_machine_details(machine_id)
                if machine_details:
                    if not gpu_type:
                        gpu_type = machine_details.gpu_type
                    if gpu_count is None:
                        gpu_count = machine_details.gpu_count
                    if hourly_rate is None:
                        hourly_rate = machine_details.hourly_rate

            # Create pod object with available details
            pod = Pod(
                id=pod_id,
                name=pod_details.get("pod_name", pod_name),
                host=ssh_host or "",
                user=ssh_user,
                port=ssh_port,
                key_path=ssh_key_path or str(Path.home() / ".ssh" / "id_rsa.pub"),
                status=pod_details.get("status", "starting"),
                hourly_rate=hourly_rate or 0.0,
                gpu_type=gpu_type or "",
                gpu_count=gpu_count or 0,
                provider_name=self.name,
            )

            return pod

        except Exception as e:
            self._logger.error(f"Error starting machine: {e}")
            print(f"Error starting machine: {e}")
            return None

    def list_pods(self) -> List[Pod]:
        """List all running pods for this provider.

        Returns:
            List of Pod objects
        """
        if not self._authenticated or not self._api_key:
            raise ValueError("Not authenticated. Call authenticate() first.")

        try:
            # Make request to Celium API to get running pods
            endpoint = "/pods"
            self._logger.debug(f"Making GET request to {endpoint}")

            response = self._make_request("GET", endpoint)
            self._logger.debug(f"Response status: {response.status_code}")

            if response.status_code != 200:
                self._logger.error(
                    f"Failed to list pods: {response.status_code} - {response.text}"
                )
                return []

            # Parse the response
            response_data = response.json()
            self._logger.debug(f"Response data: {response_data}")

            # Convert API response to Pod objects
            pods = []

            # Check if data is in content field (paginated response)
            if isinstance(response_data, dict) and "content" in response_data:
                pod_data = response_data.get("content", [])
            else:
                pod_data = response_data

            for item in pod_data:
                # Extract basic pod information
                pod_id = item.get("id", "")
                pod_name = item.get("pod_name", f"unknown-{pod_id[:8]}")

                # Extract host and port information from ports_mapping
                host = "unknown"
                port = 22
                user = "root"

                # Get executor information which contains IP address
                executor_info = item.get("executor", {})
                host = executor_info.get("executor_ip_address", "unknown")

                # Extract port mapping (SSH port 22 is mapped to an external port)
                ports_mapping = item.get("ports_mapping", {})
                for internal_port, external_port in ports_mapping.items():
                    if str(internal_port) == "22":
                        port = external_port
                        break

                # Extract GPU information
                gpu_type = item.get("gpu_name", "unknown")

                # Try to convert GPU count to integer
                gpu_count = 0
                gpu_count_str = item.get("gpu_count", "0")
                if gpu_count_str and str(gpu_count_str).isdigit():
                    gpu_count = int(gpu_count_str)

                # Extract status and normalize it
                status = item.get("status", "unknown").lower()

                # Extract price information
                hourly_rate = 0.0
                if executor_info:
                    price = executor_info.get("price_per_hour")
                    if price is not None:
                        try:
                            hourly_rate = float(price)
                        except (ValueError, TypeError):
                            hourly_rate = 0.0

                # Create Pod object with the extracted information
                pod = Pod(
                    id=pod_id,
                    name=pod_name,
                    host=host,
                    user=user,
                    port=port,
                    key_path=str(Path.home() / ".ssh" / "id_rsa.pub"),
                    status=status,
                    hourly_rate=hourly_rate,
                    gpu_type=gpu_type,
                    gpu_count=gpu_count,
                    provider_name=self.name,
                )

                pods.append(pod)

            return pods

        except Exception as e:
            self._logger.error(f"Error listing pods: {e}")
            return []

    def stop_pod(self, pod_id: str) -> tuple[bool, str]:
        """Stop a running pod.

        Args:
            pod_id: ID of the pod to stop

        Returns:
            Tuple of (success, error_message)
            - success: True if stop was successful, False otherwise
            - error_message: Error message if failed, empty string if successful
        """
        if not self._authenticated or not self._api_key:
            raise ValueError("Not authenticated. Call authenticate() first.")

        try:
            # Make request to Celium API to stop the pod
            # The endpoint is DELETE /api/executors/{pod_id}/rent
            endpoint = f"/executors/{pod_id}/rent"
            self._logger.debug(f"Making DELETE request to {endpoint}")

            response = self._make_request("DELETE", endpoint)
            self._logger.debug(f"Response status: {response.status_code}")

            if response.status_code not in [200, 202, 204]:
                error_message = response.text
                self._logger.error(
                    f"Failed to stop pod: {response.status_code} - {error_message}"
                )
                return False, error_message

            return True, ""

        except Exception as e:
            error_message = str(e)
            self._logger.error(f"Error stopping pod: {error_message}")
            return False, error_message

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        json_data: Optional[dict[str, Any]] = None,
    ) -> requests.Response:
        """Make a request to the Celium API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Optional query parameters
            json_data: Optional JSON data for POST requests

        Returns:
            Response object from the API
        """
        url = f"{self.API_BASE_URL}{endpoint}"
        headers = {
            "accept": "application/json",
            "X-API-KEY": self._api_key or "",
        }

        self._logger.debug(
            f"Making {method} request to {url} with params: {params} and json: {json_data}"
        )

        return requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_data,
            timeout=60,
        )

    def _prepare_search_params(
        self, filters: dict[str, Any], name_pattern: Optional[str]
    ) -> dict[str, Any]:
        """Prepare search parameters for the API request.

        Args:
            filters: Dictionary of filter criteria
            name_pattern: Optional pattern to filter by name

        Returns:
            Dictionary of parameters for the API request
        """
        params: dict[str, Any] = {
            "page": 0,  # Start with first page
            "size": 100,  # Get a decent amount of results
        }

        # Extract GPU filters
        gpu_config = filters.get("gpu", {})
        min_gpus = gpu_config.get("min")
        max_gpus = gpu_config.get("max")
        gpu_type = gpu_config.get("type")

        # Extract price filters
        price_config = filters.get("price", {})
        min_price = price_config.get("min")
        max_price = price_config.get("max")

        # Add GPU count filters if specified
        if min_gpus is not None:
            params["gpu_count_gte"] = min_gpus
        if max_gpus is not None:
            params["gpu_count_lte"] = max_gpus

        # Add price filters if specified
        if min_price is not None:
            params["price_per_hour_gte"] = min_price
        if max_price is not None:
            params["price_per_hour_lte"] = max_price

        # Add machine name filter if specified
        # Use partial matching with case-insensitive search
        if name_pattern is not None:
            # Filter machines locally after API response to support partial matching
            self._name_filter = name_pattern.lower()
            self._logger.debug(
                f"Will filter results for name containing: {self._name_filter}"
            )
            # Don't add to params - we'll filter results ourselves after API call

        # Handle GPU type filter the same way as name_pattern (substring matching)
        if gpu_type is not None:
            # Store GPU type for local filtering after API response
            if self._name_filter is None:  # Only set if not already set by name_pattern
                self._name_filter = gpu_type.lower()
                self._logger.debug(
                    f"Will filter results for GPU type containing: {self._name_filter}"
                )
            # Don't send GPU type to API - we'll filter locally

        return params

    def _convert_response_to_machines(
        self, data: List[dict[str, Any]]
    ) -> List[Machine]:
        """Convert API response data to Machine objects.

        Args:
            data: List of machine data from API

        Returns:
            List of Machine objects
        """
        machines: List[Machine] = []

        # Check if data is in content field (paginated response)
        if isinstance(data, dict) and "content" in data:
            self._logger.debug("Data is in paginated format with 'content' field")
            actual_data = data.get("content", [])
        else:
            actual_data = data

        self._logger.debug(f"Processing {len(actual_data)} items from API response")

        for item in actual_data:
            self._logger.debug(f"Processing machine item: {item.get('id', 'unknown')}")

            machine_id = item.get("id", "")
            machine_name = item.get("machine_name", "unknown")

            # Try to convert price_per_hour to float, default to 0.0 if not valid
            try:
                hourly_rate = float(item.get("price_per_hour", 0.0))
            except (ValueError, TypeError):
                hourly_rate = 0.0
                self._logger.debug(f"Invalid price format for machine {machine_id}")

            # Extract GPU info from specs
            specs = item.get("specs", {})

            # Try to determine GPU type and count from specs
            gpu_type, gpu_count = self._extract_gpu_info(specs, machine_name)

            machine = Machine(
                id=machine_id,
                name=machine_name,
                provider_name=self.name,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                hourly_rate=hourly_rate,
            )

            machines.append(machine)

        self._logger.debug(
            f"Converted {len(machines)} active machines from API response"
        )

        # Apply name filter if specified
        if self._name_filter is not None:
            filtered_machines = []
            for machine in machines:
                # Case-insensitive partial match
                if (
                    self._name_filter in machine.name.lower()
                    or self._name_filter in machine.gpu_type.lower()
                ):
                    filtered_machines.append(machine)

            self._logger.debug(
                f"Filtered from {len(machines)} to {len(filtered_machines)} machines based on name filter"
            )
            machines = filtered_machines

        if machines:
            self._logger.debug(f"First machine: {machines[0].to_dict()}")

        return machines

    def _extract_gpu_info(
        self, specs: dict[str, Any], machine_name: str = ""
    ) -> Tuple[str, int]:
        """Extract GPU type and count from machine specs.

        Args:
            specs: Machine specifications from API
            machine_name: The machine name to use as fallback for GPU type

        Returns:
            Tuple of (gpu_type, gpu_count)
        """
        # Default values
        gpu_type = "unknown"
        gpu_count = 0

        # Try to extract GPU information from specs
        if "gpu" in specs:
            gpu_specs = specs["gpu"]
            if isinstance(gpu_specs, dict):
                # Handle new format from sample response
                gpu_count = int(gpu_specs.get("count", 1))

                # Try to get GPU details from the details list if available
                if (
                    "details" in gpu_specs
                    and isinstance(gpu_specs["details"], list)
                    and gpu_specs["details"]
                ):
                    # Use the model name from the first GPU detail
                    first_gpu = gpu_specs["details"][0]
                    if isinstance(first_gpu, dict) and "model" in first_gpu:
                        gpu_type = first_gpu["model"]

                # If no details found, fallback to alternative sources
                if gpu_type == "unknown":
                    # Try to extract from machine_name if it looks like a GPU model
                    if machine_name and any(
                        gpu in machine_name.lower()
                        for gpu in [
                            "nvidia",
                            "rtx",
                            "gtx",
                            "a100",
                            "h100",
                            "v100",
                            "t4",
                            "l40",
                            "a6000",
                            "a40",
                        ]
                    ):
                        gpu_type = machine_name
            elif isinstance(gpu_specs, list):
                # If specs.gpu is a list of GPUs
                if gpu_specs:
                    # Take type from first GPU
                    gpu_type = gpu_specs[0].get(
                        "name", gpu_specs[0].get("model", "unknown")
                    )
                    # Count is the length of the list
                    gpu_count = len(gpu_specs)

        return gpu_type, gpu_count
