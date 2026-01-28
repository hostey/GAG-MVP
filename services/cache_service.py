# services/cache_service.py
import time
import pickle
import redis
from typing import Any, Optional, Dict
from datetime import timedelta
import hashlib
import json

from config.settings import settings
from core.utils.logger import get_logger
from core.utils.serializers import CacheSerializer

logger = get_logger(__name__)


class CacheService:
    """High-performance caching service with Redis and memory fallback."""

    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }

        # Initialize Redis if enabled
        if settings.CACHE_ENABLED:
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=False,
                    socket_timeout=5,
                    retry_on_timeout=True
                )

                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache initialized successfully")

            except redis.ConnectionError as e:
                logger.warning(f"Redis not available, using in-memory cache: {e}")
                self.redis_client = None

        logger.info(f"Cache service initialized (Redis: {self.redis_client is not None})")

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        # Try Redis first
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value is not None:
                    self.stats['hits'] += 1
                    return pickle.loads(value)
            except redis.RedisError as e:
                logger.warning(f"Redis get failed: {e}")

        # Try memory cache
        if key in self.memory_cache:
            item = self.memory_cache[key]
            if time.time() < item['expiry']:
                self.stats['hits'] += 1
                return item['value']
            else:
                del self.memory_cache[key]

        self.stats['misses'] += 1
        return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        if ttl is None:
            ttl = settings.CACHE_TTL

        serialized = pickle.dumps(value)

        # Try Redis first
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, serialized)
                self.stats['sets'] += 1
                return
            except redis.RedisError as e:
                logger.warning(f"Redis set failed: {e}")

        # Fallback to memory cache
        self.memory_cache[key] = {
            'value': value,
            'expiry': time.time() + ttl
        }
        self.stats['sets'] += 1

    def delete(self, key: str):
        """Delete key from cache."""
        # Delete from Redis
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except redis.RedisError:
                pass

        # Delete from memory
        if key in self.memory_cache:
            del self.memory_cache[key]

        self.stats['deletes'] += 1

    def clear(self, pattern: Optional[str] = None):
        """Clear cache entries."""
        if pattern:
            # Clear matching keys
            if self.redis_client:
                try:
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                except redis.RedisError:
                    pass

            # Clear memory cache
            for key in list(self.memory_cache.keys()):
                if pattern in key:
                    del self.memory_cache[key]
        else:
            # Clear all
            if self.redis_client:
                try:
                    self.redis_client.flushdb()
                except redis.RedisError:
                    pass

            self.memory_cache.clear()

        logger.info("Cache cleared")

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if self.redis_client:
            try:
                return bool(self.redis_client.exists(key))
            except redis.RedisError:
                pass

        return key in self.memory_cache

    def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter in cache."""
        if self.redis_client:
            try:
                return self.redis_client.incrby(key, amount)
            except redis.RedisError:
                pass

        # Memory cache implementation
        current = self.get(key, 0)
        new_value = current + amount
        self.set(key, new_value)
        return new_value

    def get_or_set(self, key: str, factory: callable, ttl: Optional[int] = None) -> Any:
        """Get value from cache or set it using factory function."""
        value = self.get(key)
        if value is not None:
            return value

        value = factory()
        self.set(key, value, ttl)
        return value

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset cache statistics."""
        self.stats = {'hits': 0, 'misses': 0, 'sets': 0, 'deletes': 0}

    def generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_parts = []

        # Add positional arguments
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(hashlib.md5(pickle.dumps(arg)).hexdigest())

        # Add keyword arguments
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        return hashlib.md5(":".join(key_parts).encode()).hexdigest()


class SimulationCache:
    """Specialized cache for simulation results."""

    def __init__(self, cache_service: CacheService):
        self.cache = cache_service
        self.prefix = "simulation"

    def get_result(self, parameters: Dict[str, Any]) -> Optional[Any]:
        """Get cached simulation result for parameters."""
        cache_key = self._generate_parameter_key(parameters)
        return self.cache.get(cache_key)

    def set_result(self, parameters: Dict[str, Any], result: Any, ttl: Optional[int] = None):
        """Cache simulation result."""
        cache_key = self._generate_parameter_key(parameters)
        self.cache.set(cache_key, result, ttl)

    def get_training_history(self, model_id: str) -> Optional[Dict]:
        """Get cached training history for model."""
        cache_key = f"{self.prefix}:training:{model_id}"
        return self.cache.get(cache_key)

    def set_training_history(self, model_id: str, history: Dict, ttl: int = 86400):
        """Cache training history."""
        cache_key = f"{self.prefix}:training:{model_id}"
        self.cache.set(cache_key, history, ttl)

    def get_metrics(self, simulation_id: str) -> Optional[Dict]:
        """Get cached metrics for simulation."""
        cache_key = f"{self.prefix}:metrics:{simulation_id}"
        return self.cache.get(cache_key)

    def set_metrics(self, simulation_id: str, metrics: Dict, ttl: int = 3600):
        """Cache simulation metrics."""
        cache_key = f"{self.prefix}:metrics:{simulation_id}"
        self.cache.set(cache_key, metrics, ttl)

    def clear_simulation_cache(self, simulation_id: str):
        """Clear all cache for a simulation."""
        pattern = f"{self.prefix}:*:{simulation_id}"
        self.cache.clear(pattern)

    def _generate_parameter_key(self, parameters: Dict[str, Any]) -> str:
        """Generate cache key from simulation parameters."""
        return self.cache.generate_key(self.prefix, "result", **parameters)