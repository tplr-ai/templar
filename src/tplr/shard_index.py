class ShardIndex:
    def __init__(self, shard_data):
        """
        Initialize shard index with precomputed cumulative row counts
        for efficient binary search operations.

        Args:
            shard_data (dict): The loaded _shard_sizes.json data
        """
        self.shard_data = shard_data
        self.indices = {}

        for config_name, config_data in shard_data.items():
            self._build_index(config_name, config_data)

    def _build_index(self, config_name, config_data):
        """
        Build cumulative index for a specific configuration.

        Args:
            config_name (str): The configuration name
            config_data (dict): The configuration metadata
        """
        shards = config_data.get("shards", [])
        if not shards:
            self.indices[config_name] = {"cum_rows": [], "shards": []}
            return

        cum_rows = []
        start_row = 0

        for i, shard in enumerate(shards):
            cum_rows.append(start_row)
            start_row += shard["num_rows"]

        cum_rows.append(start_row)

        self.indices[config_name] = {
            "cum_rows": cum_rows,
            "shards": shards,
            "total_rows": config_data.get("total_rows", start_row),
        }

    def find_shard(self, config_name, page_number):
        """
        Find the appropriate shard for a given page number using binary search.

        Args:
            config_name (str): The configuration name
            page_number (int): The global page number to locate

        Returns:
            tuple: (shard, shard_offset, shard_index) where:
                - shard is the shard descriptor dict
                - shard_offset is the local offset within the shard
                - shard_index is the index of the shard in the original array

        Raises:
            ValueError: If the page_number is out of bounds or config doesn't exist
        """
        index_data = self.indices.get(config_name)
        if not index_data:
            raise ValueError(f"No index found for config '{config_name}'")

        cum_rows = index_data["cum_rows"]
        shards = index_data["shards"]

        if not cum_rows or page_number >= cum_rows[-1]:
            raise ValueError(
                f"Page {page_number} out of bounds for config '{config_name}'"
            )

        # Binary search to find the shard containing the page
        left, right = 0, len(cum_rows) - 2  # -2 because the last entry is the boundary

        while left <= right:
            mid = (left + right) // 2

            # Check if page_number is in this shard's range
            start_row = cum_rows[mid]
            end_row = cum_rows[mid + 1]

            if start_row <= page_number < end_row:
                # Found the correct shard
                shard = shards[mid]
                shard_offset = page_number - start_row
                return shard, shard_offset, mid
            elif page_number < start_row:
                right = mid - 1
            else:
                left = mid + 1

        # This should never happen if our binary search is implemented correctly
        raise ValueError(
            f"Binary search failed for page {page_number} in config '{config_name}'"
        )
