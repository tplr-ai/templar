import threading
import time
import hashlib


class GradientCache:
    """
    A process-local LRU cache for gradient files.
    Multiple instances will share the same class-level cache.
    """

    # Class-level variables shared across all instances
    _cache = {}  # key -> (data, timestamp)
    _cache_lock = threading.Lock()
    _max_size = 200  # Maximum number of items in cache
    _max_age = 450  # Maximum age in seconds (1 hour)
    _hits = 0
    _misses = 0

    def __init__(self, logger=None):
        """Initialize with optional logger."""
        self.logger = logger

    @classmethod
    def set_max_size(cls, size):
        """Set maximum cache size."""
        cls._max_size = size

    @classmethod
    def set_max_age(cls, age):
        """Set maximum item age in seconds."""
        cls._max_age = age

    def _make_key(self, uid, window, key, time_min, time_max):
        """Create a unique cache key."""
        components = f"{uid}_{window}_{key}_{time_min}_{time_max}"
        return hashlib.md5(components.encode()).hexdigest()

    def get(self, uid, window, key, time_min=None, time_max=None):
        """
        Try to get data from cache.

        Returns:
            The cached data or None if not found
        """
        cache_key = self._make_key(uid, window, key, time_min, time_max)

        with self._cache_lock:
            if cache_key in self._cache:
                data, timestamp = self._cache[cache_key]

                # Check if item is expired
                if time.time() - timestamp > self._max_age:
                    # Remove expired item
                    del self._cache[cache_key]
                    self._misses += 1
                    return None

                # Update timestamp for LRU tracking
                self._cache[cache_key] = (data, time.time())
                self._hits += 1

                if self.logger:
                    self.logger.debug(
                        f"Cache hit for UID {uid} (hits: {self._hits}, misses: {self._misses})"
                    )

                return data

            self._misses += 1
            return None

    def put(self, uid, window, key, data, time_min=None, time_max=None):
        """
        Store data in cache.

        Returns:
            True if stored successfully
        """
        cache_key = self._make_key(uid, window, key, time_min, time_max)

        with self._cache_lock:
            # Make room if needed
            if len(self._cache) >= self._max_size:
                # Sort by timestamp (oldest first)
                sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])
                # Remove oldest 10% of items
                for old_key, _ in sorted_items[: max(1, len(sorted_items) // 10)]:
                    del self._cache[old_key]

            # Store with current timestamp
            self._cache[cache_key] = (data, time.time())

            if self.logger:
                self.logger.debug(
                    f"Cached data for UID {uid}, cache size: {len(self._cache)}"
                )

            return True
