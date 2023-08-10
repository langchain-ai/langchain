from typing import Any, Iterator, List, Optional, Sequence, Tuple, cast

from langchain.schema import BaseStore


class RedisStore(BaseStore[str, bytes]):
    """BaseStore implementation using Redis as the underlying store.

    Examples:
        Create a RedisStore instance and perform operations on it:

        .. code-block:: python

            # Instantiate the RedisStore with a Redis connection
            from langchain.storage import RedisStore
            from langchain.vectorstores.redis import get_client

            client = get_client('redis://localhost:6379')
            redis_store = RedisStore(client)

            # Set values for keys
            redis_store.mset([("key1", b"value1"), ("key2", b"value2")])

            # Get values for keys
            values = redis_store.mget(["key1", "key2"])
            # [b"value1", b"value2"]

            # Delete keys
            redis_store.mdelete(["key1"])

            # Iterate over keys
            for key in redis_store.yield_keys():
                print(key)
    """

    def __init__(
        self, client: Any, *, ttl: Optional[int] = None, namespace: Optional[str] = None
    ) -> None:
        """Initialize the RedisStore with a Redis connection.

        Args:
            client: A Redis connection instance
            ttl: time to expire keys in seconds if provided,
                 if None keys will never expire
            namespace: if provided, all keys will be prefixed with this namespace
        """
        try:
            from redis import Redis
        except ImportError as e:
            raise ImportError(
                "The RedisStore requires the redis library to be installed. "
                "pip install redis"
            ) from e

        if not isinstance(client, Redis):
            raise TypeError(
                f"Expected Redis client, got {type(client).__name__} instead."
            )

        self.client = client

        if not isinstance(ttl, int) and ttl is not None:
            raise TypeError(f"Expected int or None, got {type(ttl)} instead.")

        self.ttl = ttl
        self.namespace = namespace
        self.namespace_delimiter = "/"

    def _get_prefixed_key(self, key: str) -> str:
        """Get the key with the namespace prefix.

        Args:
            key (str): The original key.

        Returns:
            str: The key with the namespace prefix.
        """
        if self.namespace:
            return f"{self.namespace}{self.namespace_delimiter}{key}"
        return key

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        """Get the values associated with the given keys."""
        return cast(
            List[Optional[bytes]],
            self.client.mget([self._get_prefixed_key(key) for key in keys]),
        )

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        """Set the given key-value pairs."""
        pipe = self.client.pipeline()

        for key, value in key_value_pairs:
            pipe.set(self._get_prefixed_key(key), value, ex=self.ttl)
        pipe.execute()

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
        scan_iter = cast(Iterator[bytes], self.client.scan_iter(match=pattern))
        for key in scan_iter:
            decoded_key = key.decode("utf-8")
            if self.namespace:
                relative_key = decoded_key[len(self.namespace) + 1 :]
                yield relative_key
            else:
                yield decoded_key
