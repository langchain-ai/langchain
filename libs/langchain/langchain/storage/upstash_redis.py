from typing import Any, Iterator, List, Optional, Sequence, Tuple, cast

from langchain.schema import BaseStore


class UpstashRedisStore(BaseStore[str, str]):
    """BaseStore implementation using Upstash Redis as the underlying store."""

    def __init__(
        self,
        *,
        client: Any = None,
        url: Optional[str] = None,
        token: Optional[str] = None,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """Initialize the UpstashRedisStore with HTTP API.

        Must provide either an Upstash Redis client or a url.

        Args:
            client: An Upstash Redis instance
            url: UPSTASH_REDIS_REST_URL
            token: UPSTASH_REDIS_REST_TOKEN
            ttl: time to expire keys in seconds if provided,
                 if None keys will never expire
            namespace: if provided, all keys will be prefixed with this namespace
        """
        try:
            from upstash_redis import Redis
        except ImportError as e:
            raise ImportError(
                "UpstashRedisStore requires the upstash_redis library to be installed. "
                "pip install upstash_redis"
            ) from e

        if client and url:
            raise ValueError(
                "Either an Upstash Redis client or a url must be provided, not both."
            )

        if client:
            if not isinstance(client, Redis):
                raise TypeError(
                    f"Expected Upstash Redis client, got {type(client).__name__}."
                )
            _client = client
        else:
            if not url or not token:
                raise ValueError(
                    "Either an Upstash Redis client or url and token must be provided."
                )
            _client = Redis(url=url, token=token)

        self.client = _client

        if not isinstance(ttl, int) and ttl is not None:
            raise TypeError(f"Expected int or None, got {type(ttl)} instead.")

        self.ttl = ttl
        self.namespace = namespace

    def _get_prefixed_key(self, key: str) -> str:
        """Get the key with the namespace prefix.

        Args:
            key (str): The original key.

        Returns:
            str: The key with the namespace prefix.
        """
        delimiter = "/"
        if self.namespace:
            return f"{self.namespace}{delimiter}{key}"
        return key

    def mget(self, keys: Sequence[str]) -> List[Optional[str]]:
        """Get the values associated with the given keys."""

        keys = [self._get_prefixed_key(key) for key in keys]
        return cast(
            List[Optional[str]],
            self.client.mget(*keys),
        )

    def mset(self, key_value_pairs: Sequence[Tuple[str, str]]) -> None:
        """Set the given key-value pairs."""
        for key, value in key_value_pairs:
            self.client.set(self._get_prefixed_key(key), value, ex=self.ttl)

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys."""
        _keys = [self._get_prefixed_key(key) for key in keys]
        self.client.delete(*_keys)

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        """Yield keys in the store."""
        if prefix:
            pattern = self._get_prefixed_key(prefix)
        else:
            pattern = self._get_prefixed_key("*")

        cursor, keys = self.client.scan(0, match=pattern)
        for key in keys:
            if self.namespace:
                relative_key = key[len(self.namespace) + 1 :]
                yield relative_key
            else:
                yield key

        while cursor != 0:
            cursor, keys = self.client.scan(cursor, match=pattern)
            for key in keys:
                if self.namespace:
                    relative_key = key[len(self.namespace) + 1 :]
                    yield relative_key
                else:
                    yield key
