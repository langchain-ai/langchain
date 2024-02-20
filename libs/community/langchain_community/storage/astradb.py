from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from langchain_core.stores import BaseStore, ByteStore

from langchain_community.utilities.astradb import (
    SetupMode,
    _AstraDBCollectionEnvironment,
)

if TYPE_CHECKING:
    from astrapy.db import AstraDB, AsyncAstraDB

V = TypeVar("V")


class AstraDBBaseStore(Generic[V], BaseStore[str, V], ABC):
    """Base class for the DataStax AstraDB data store."""

    def __init__(
        self,
        collection_name: str,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_client: Optional[AstraDB] = None,
        namespace: Optional[str] = None,
        *,
        async_astra_db_client: Optional[AsyncAstraDB] = None,
        pre_delete_collection: bool = False,
        setup_mode: SetupMode = SetupMode.SYNC,
    ) -> None:
        self.astra_env = _AstraDBCollectionEnvironment(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            astra_db_client=astra_db_client,
            async_astra_db_client=async_astra_db_client,
            namespace=namespace,
            setup_mode=setup_mode,
            pre_delete_collection=pre_delete_collection,
        )
        self.collection = self.astra_env.collection
        self.async_collection = self.astra_env.async_collection

    @abstractmethod
    def decode_value(self, value: Any) -> Optional[V]:
        """Decodes value from Astra DB"""

    @abstractmethod
    def encode_value(self, value: Optional[V]) -> Any:
        """Encodes value for Astra DB"""

    def mget(self, keys: Sequence[str]) -> List[Optional[V]]:
        """Get the values associated with the given keys."""
        self.astra_env.ensure_db_setup()
        docs_dict = {}
        for doc in self.collection.paginated_find(filter={"_id": {"$in": list(keys)}}):
            docs_dict[doc["_id"]] = doc.get("value")
        return [self.decode_value(docs_dict.get(key)) for key in keys]

    async def amget(self, keys: Sequence[str]) -> List[Optional[V]]:
        """Get the values associated with the given keys."""
        await self.astra_env.aensure_db_setup()
        docs_dict = {}
        async for doc in self.async_collection.paginated_find(
            filter={"_id": {"$in": list(keys)}}
        ):
            docs_dict[doc["_id"]] = doc.get("value")
        return [self.decode_value(docs_dict.get(key)) for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        """Set the given key-value pairs."""
        self.astra_env.ensure_db_setup()
        for k, v in key_value_pairs:
            self.collection.upsert({"_id": k, "value": self.encode_value(v)})

    async def amset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        """Set the given key-value pairs."""
        await self.astra_env.aensure_db_setup()
        for k, v in key_value_pairs:
            await self.async_collection.upsert(
                {"_id": k, "value": self.encode_value(v)}
            )

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys."""
        self.astra_env.ensure_db_setup()
        self.collection.delete_many(filter={"_id": {"$in": list(keys)}})

    async def amdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys."""
        await self.astra_env.aensure_db_setup()
        await self.async_collection.delete_many(filter={"_id": {"$in": list(keys)}})

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        """Yield keys in the store."""
        self.astra_env.ensure_db_setup()
        docs = self.collection.paginated_find()
        for doc in docs:
            key = doc["_id"]
            if not prefix or key.startswith(prefix):
                yield key

    async def ayield_keys(self, *, prefix: Optional[str] = None) -> AsyncIterator[str]:
        """Yield keys in the store."""
        await self.astra_env.aensure_db_setup()
        async for doc in self.async_collection.paginated_find():
            key = doc["_id"]
            if not prefix or key.startswith(prefix):
                yield key


class AstraDBStore(AstraDBBaseStore[Any]):
    """BaseStore implementation using DataStax AstraDB as the underlying store.

    The value type can be any type serializable by json.dumps.
    Can be used to store embeddings with the CacheBackedEmbeddings.
    Documents in the AstraDB collection will have the format
    {
      "_id": "<key>",
      "value": <value>
    }
    """

    def decode_value(self, value: Any) -> Any:
        return value

    def encode_value(self, value: Any) -> Any:
        return value


class AstraDBByteStore(AstraDBBaseStore[bytes], ByteStore):
    """ByteStore implementation using DataStax AstraDB as the underlying store.

    The bytes values are converted to base64 encoded strings
    Documents in the AstraDB collection will have the format
    {
      "_id": "<key>",
      "value": "<byte64 string value>"
    }
    """

    def decode_value(self, value: Any) -> Optional[bytes]:
        if value is None:
            return None
        return base64.b64decode(value)

    def encode_value(self, value: Optional[bytes]) -> Any:
        if value is None:
            return None
        return base64.b64encode(value).decode("ascii")
