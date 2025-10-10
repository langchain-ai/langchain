from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import (
    Any,
    TypeVar,
)

from langchain_core.stores import BaseStore

K = TypeVar("K")
V = TypeVar("V")


class EncoderBackedStore(BaseStore[K, V]):
    """Wraps a store with key and value encoders/decoders.

    Examples that uses JSON for encoding/decoding:

    ```python
    import json


    def key_encoder(key: int) -> str:
        return json.dumps(key)


    def value_serializer(value: float) -> str:
        return json.dumps(value)


    def value_deserializer(serialized_value: str) -> float:
        return json.loads(serialized_value)


    # Create an instance of the abstract store
    abstract_store = MyCustomStore()

    # Create an instance of the encoder-backed store
    store = EncoderBackedStore(
        store=abstract_store,
        key_encoder=key_encoder,
        value_serializer=value_serializer,
        value_deserializer=value_deserializer,
    )

    # Use the encoder-backed store methods
    store.mset([(1, 3.14), (2, 2.718)])
    values = store.mget([1, 2])  # Retrieves [3.14, 2.718]
    store.mdelete([1, 2])  # Deletes the keys 1 and 2
    ```
    """

    def __init__(
        self,
        store: BaseStore[str, Any],
        key_encoder: Callable[[K], str],
        value_serializer: Callable[[V], bytes],
        value_deserializer: Callable[[Any], V],
    ) -> None:
        """Initialize an EncodedStore."""
        self.store = store
        self.key_encoder = key_encoder
        self.value_serializer = value_serializer
        self.value_deserializer = value_deserializer

    def mget(self, keys: Sequence[K]) -> list[V | None]:
        """Get the values associated with the given keys."""
        encoded_keys: list[str] = [self.key_encoder(key) for key in keys]
        values = self.store.mget(encoded_keys)
        return [
            self.value_deserializer(value) if value is not None else value
            for value in values
        ]

    async def amget(self, keys: Sequence[K]) -> list[V | None]:
        """Get the values associated with the given keys."""
        encoded_keys: list[str] = [self.key_encoder(key) for key in keys]
        values = await self.store.amget(encoded_keys)
        return [
            self.value_deserializer(value) if value is not None else value
            for value in values
        ]

    def mset(self, key_value_pairs: Sequence[tuple[K, V]]) -> None:
        """Set the values for the given keys."""
        encoded_pairs = [
            (self.key_encoder(key), self.value_serializer(value))
            for key, value in key_value_pairs
        ]
        self.store.mset(encoded_pairs)

    async def amset(self, key_value_pairs: Sequence[tuple[K, V]]) -> None:
        """Set the values for the given keys."""
        encoded_pairs = [
            (self.key_encoder(key), self.value_serializer(value))
            for key, value in key_value_pairs
        ]
        await self.store.amset(encoded_pairs)

    def mdelete(self, keys: Sequence[K]) -> None:
        """Delete the given keys and their associated values."""
        encoded_keys = [self.key_encoder(key) for key in keys]
        self.store.mdelete(encoded_keys)

    async def amdelete(self, keys: Sequence[K]) -> None:
        """Delete the given keys and their associated values."""
        encoded_keys = [self.key_encoder(key) for key in keys]
        await self.store.amdelete(encoded_keys)

    def yield_keys(
        self,
        *,
        prefix: str | None = None,
    ) -> Iterator[K] | Iterator[str]:
        """Get an iterator over keys that match the given prefix."""
        # For the time being this does not return K, but str
        # it's for debugging purposes. Should fix this.
        yield from self.store.yield_keys(prefix=prefix)

    async def ayield_keys(
        self,
        *,
        prefix: str | None = None,
    ) -> AsyncIterator[K] | AsyncIterator[str]:
        """Get an iterator over keys that match the given prefix."""
        # For the time being this does not return K, but str
        # it's for debugging purposes. Should fix this.
        async for key in self.store.ayield_keys(prefix=prefix):
            yield key
