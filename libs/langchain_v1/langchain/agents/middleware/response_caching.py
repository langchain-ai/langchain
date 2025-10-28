"""Response caching middleware for agents.

This middleware caches model responses to avoid redundant API calls for
identical or similar requests, reducing costs and improving response times.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable


@dataclass
class CacheEntry:
    """A cached model response with metadata.

    Attributes:
        response: The cached model response.
        timestamp: Unix timestamp when the entry was created.
        hits: Number of times this cache entry has been used.
    """

    response: ModelResponse
    timestamp: float
    hits: int = 0


class ResponseCachingMiddleware(AgentMiddleware):
    """Middleware that caches model responses to reduce redundant API calls.

    This middleware computes a fingerprint for each model request and caches
    the response. When a subsequent request has the same fingerprint, the cached
    response is returned instead of making a new API call.

    The cache supports:
    - TTL (time-to-live) for automatic cache expiration
    - Maximum cache size with LRU (least recently used) eviction
    - Customizable cache key generation

    Example:
        Basic usage with default settings:
        ```python
        from langchain.agents.middleware import ResponseCachingMiddleware
        from langchain.agents import create_agent

        cache = ResponseCachingMiddleware(ttl=3600, max_size=100)
        agent = create_agent("openai:gpt-4o", middleware=[cache])

        # First call hits the API
        result1 = agent.invoke({"messages": [HumanMessage("What is 2+2?")]})

        # Second identical call returns cached response
        result2 = agent.invoke({"messages": [HumanMessage("What is 2+2?")]})
        ```

        Custom cache key function:
        ```python
        def custom_key(request: ModelRequest) -> str:
            # Only cache based on the last user message
            last_msg = request.messages[-1] if request.messages else ""
            return str(last_msg.content if hasattr(last_msg, "content") else last_msg)


        cache = ResponseCachingMiddleware(
            ttl=1800,
            max_size=50,
            cache_key_fn=custom_key,
        )
        ```

        Disable TTL for permanent caching:
        ```python
        cache = ResponseCachingMiddleware(ttl=None, max_size=1000)
        ```
    """

    def __init__(
        self,
        *,
        ttl: float | None = 3600,
        max_size: int = 100,
        cache_key_fn: Callable[[ModelRequest], str] | None = None,
    ) -> None:
        """Initialize the response caching middleware.

        Args:
            ttl: Time-to-live for cache entries in seconds. After this time,
                cached responses expire and are removed. Set to `None` to disable
                expiration. Defaults to 3600 seconds (1 hour).
            max_size: Maximum number of entries to store in the cache. When this
                limit is reached, the least recently used entry is evicted.
                Must be at least 1. Defaults to 100.
            cache_key_fn: Optional custom function to generate cache keys from
                requests. If `None`, uses the default fingerprinting based on
                messages, system prompt, and model settings. The function should
                take a `ModelRequest` and return a string key. Defaults to `None`.

        Raises:
            ValueError: If `max_size` is less than 1 or if `ttl` is negative.
        """
        super().__init__()

        if max_size < 1:
            msg = f"max_size must be at least 1, got {max_size}"
            raise ValueError(msg)

        if ttl is not None and ttl < 0:
            msg = f"ttl must be non-negative or None, got {ttl}"
            raise ValueError(msg)

        self.ttl = ttl
        self.max_size = max_size
        self.cache_key_fn = cache_key_fn or self._default_cache_key
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def _default_cache_key(self, request: ModelRequest) -> str:
        """Generate a cache key fingerprint from the request.

        The fingerprint includes:
        - Model class name (not instance, as instances may vary)
        - System prompt
        - All messages (content and type)
        - Model settings (temperature, etc.)

        Args:
            request: The model request to fingerprint.

        Returns:
            A SHA256 hash string representing the request fingerprint.
        """
        # Create a deterministic representation of the request
        fingerprint_data: dict[str, Any] = {
            "model_class": request.model.__class__.__name__,
            "system_prompt": request.system_prompt,
            "messages": self._serialize_messages(request.messages),
            "model_settings": request.model_settings,
        }

        # Convert to JSON string (sorted keys for consistency)
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)

        # Return SHA256 hash
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()

    def _serialize_messages(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Serialize messages to a deterministic format for fingerprinting.

        Args:
            messages: List of messages to serialize.

        Returns:
            List of dictionaries representing the messages.
        """
        serialized = []
        for msg in messages:
            msg_dict: dict[str, Any] = {
                "type": msg.__class__.__name__,
                "content": msg.content,
            }

            # Include tool calls if present (for AIMessage)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls

            # Include tool call ID if present (for ToolMessage)
            if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id

            serialized.append(msg_dict)

        return serialized

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired.

        Args:
            entry: The cache entry to check.

        Returns:
            `True` if the entry has expired, `False` otherwise.
        """
        if self.ttl is None:
            return False

        current_time = time.time()
        return (current_time - entry.timestamp) > self.ttl

    def _evict_expired(self) -> None:
        """Remove all expired entries from the cache."""
        if self.ttl is None:
            return

        expired_keys = [key for key, entry in self._cache.items() if self._is_expired(entry)]

        for key in expired_keys:
            del self._cache[key]

    def _evict_lru(self) -> None:
        """Remove the least recently used entry to make space."""
        if self._cache:
            # OrderedDict removes from the beginning (oldest)
            self._cache.popitem(last=False)

    def _get_from_cache(self, cache_key: str) -> ModelResponse | None:
        """Retrieve a response from the cache if available and valid.

        Args:
            cache_key: The cache key to look up.

        Returns:
            The cached `ModelResponse` if found and not expired, `None` otherwise.
        """
        # First, clean up expired entries
        self._evict_expired()

        # Check if key exists and is not expired
        if cache_key in self._cache:
            entry = self._cache[cache_key]

            if not self._is_expired(entry):
                # Move to end (mark as recently used)
                self._cache.move_to_end(cache_key)
                entry.hits += 1
                return entry.response

            # Entry is expired, remove it
            del self._cache[cache_key]

        return None

    def _add_to_cache(self, cache_key: str, response: ModelResponse) -> None:
        """Add a response to the cache.

        Args:
            cache_key: The cache key for this response.
            response: The model response to cache.
        """
        # If key already exists, remove it first (will be re-added)
        if cache_key in self._cache:
            del self._cache[cache_key]

        # If at max size, evict LRU entry
        if len(self._cache) >= self.max_size:
            self._evict_lru()

        # Add new entry
        entry = CacheEntry(
            response=response,
            timestamp=time.time(),
            hits=0,
        )
        self._cache[cache_key] = entry

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the current cache state.

        Returns:
            Dictionary containing cache statistics:
            - `size`: Current number of entries in the cache
            - `max_size`: Maximum cache size
            - `ttl`: Time-to-live setting
            - `total_hits`: Total number of cache hits across all entries
            - `entries`: List of cache entry information (key, hits, age)
        """
        self._evict_expired()

        current_time = time.time()
        entries_info = []

        for key, entry in self._cache.items():
            entries_info.append(
                {
                    "key": key[:16] + "...",  # Truncate for readability
                    "hits": entry.hits,
                    "age_seconds": current_time - entry.timestamp,
                }
            )

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "total_hits": sum(e.hits for e in self._cache.values()),
            "entries": entries_info,
        }

    def clear_cache(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Intercept model calls to implement caching.

        Args:
            request: The model request.
            handler: Callback to execute the model call.

        Returns:
            Cached `ModelResponse` if available, otherwise executes the handler
            and caches the result.
        """
        # Generate cache key
        cache_key = self.cache_key_fn(request)

        # Try to get from cache
        cached_response = self._get_from_cache(cache_key)
        if cached_response is not None:
            return cached_response

        # Cache miss - execute the handler
        response = handler(request)

        # Cache the response
        self._add_to_cache(cache_key, response)

        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Intercept async model calls to implement caching.

        Args:
            request: The model request.
            handler: Async callback to execute the model call.

        Returns:
            Cached `ModelResponse` if available, otherwise executes the handler
            and caches the result.
        """
        # Generate cache key
        cache_key = self.cache_key_fn(request)

        # Try to get from cache
        cached_response = self._get_from_cache(cache_key)
        if cached_response is not None:
            return cached_response

        # Cache miss - execute the handler
        response = await handler(request)

        # Cache the response
        self._add_to_cache(cache_key, response)

        return response
