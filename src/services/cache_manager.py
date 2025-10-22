"""
Cache manager for scoped, predictable caching behavior.
"""

from typing import Optional, Dict, Any, Callable, Pattern
from datetime import datetime, timedelta
import hashlib
import functools
import logging
import re

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Request-scoped cache manager with predictable behavior.
    Replaces global process-level caching with controlled TTL and invalidation.
    """

    def __init__(self, default_ttl: int = 300):
        """
        Initialize the cache manager.

        Args:
            default_ttl: Default time-to-live in seconds (default: 5 minutes)
        """
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self.default_ttl = default_ttl

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable,
        ttl: Optional[int] = None,
        *args,
        **kwargs
    ) -> Any:
        """
        Get cached value or compute and cache it.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl: Time-to-live in seconds (uses default if None)
            *args: Arguments to pass to compute_fn
            **kwargs: Keyword arguments to pass to compute_fn

        Returns:
            Cached or computed value
        """
        ttl = ttl if ttl is not None else self.default_ttl

        # Check if cached and not expired
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=ttl):
                logger.debug(f"Cache hit for key: {key}")
                return value
            else:
                logger.debug(f"Cache expired for key: {key}")
                del self._cache[key]

        # Compute and cache
        logger.debug(f"Cache miss for key: {key}, computing...")
        value = compute_fn(*args, **kwargs)
        self._cache[key] = (value, datetime.now())
        return value

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value without computing.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.default_ttl):
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any):
        """
        Set cached value.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = (value, datetime.now())
        logger.debug(f"Cached value for key: {key}")

    def invalidate(self, pattern: Optional[str] = None):
        """
        Invalidate cache entries.

        Args:
            pattern: Regex pattern to match keys (invalidates all if None)
        """
        if pattern is None:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Invalidated all {count} cache entries")
        else:
            regex = re.compile(pattern)
            keys_to_remove = [k for k in self._cache.keys() if regex.search(k)]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching pattern: {pattern}")

    def clear_expired(self):
        """Remove all expired cache entries."""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if now - timestamp >= timedelta(seconds=self.default_ttl)
        ]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")

    def make_key(self, *args, **kwargs) -> str:
        """
        Create a cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            MD5 hash of serialized arguments
        """
        key_data = str((args, sorted(kwargs.items())))
        return hashlib.md5(key_data.encode()).hexdigest()

    def cached(self, ttl: Optional[int] = None):
        """
        Decorator for caching function results.

        Args:
            ttl: Time-to-live in seconds

        Example:
            @cache_manager.cached(ttl=60)
            def expensive_function(param):
                return compute_something(param)
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key from function name and arguments
                key_data = (func.__name__, args, sorted(kwargs.items()))
                key = hashlib.md5(str(key_data).encode()).hexdigest()
                return self.get_or_compute(key, func, ttl, *args, **kwargs)
            return wrapper
        return decorator

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        now = datetime.now()
        expired_count = sum(
            1 for _, (_, timestamp) in self._cache.items()
            if now - timestamp >= timedelta(seconds=self.default_ttl)
        )
        return {
            'total_entries': len(self._cache),
            'expired_entries': expired_count,
            'active_entries': len(self._cache) - expired_count,
            'default_ttl': self.default_ttl
        }
