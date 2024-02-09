from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from langchain_core.stores import BaseStore, ByteStore

from langchain_community.utilities.astradb import AstraDBEnvironment

if TYPE_CHECKING:
    from astrapy.db import AstraDB

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
    ) -> None:
        astra_env = AstraDBEnvironment(
            token=token,
            api_endpoint=api_endpoint,
            astra_db_client=astra_db_client,
            namespace=namespace,
        )
        self.astra_db = astra_env.astra_db
        self.collection = self.astra_db.create_collection(
            collection_name=collection_name,
        )

    @abstractmethod
    def decode_value(self, value: Any) -> Optional[V]:
        """Decodes value from Astra DB"""

    @abstractmethod
    def encode_value(self, value: Optional[V]) -> Any:
        """Encodes value for Astra DB"""

    def mget(self, keys: Sequence[str]) -> List[Optional[V]]:
        """Get the values associated with the given keys."""
        docs_dict = {}
        for doc in self.collection.paginated_find(filter={"_id": {"$in": list(keys)}}):
            docs_dict[doc["_id"]] = doc.get("value")
        return [self.decode_value(docs_dict.get(key)) for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        """Set the given key-value pairs."""
        for k, v in key_value_pairs:
            self.collection.upsert({"_id": k, "value": self.encode_value(v)})

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys."""
        self.collection.delete_many(filter={"_id": {"$in": list(keys)}})

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        """Yield keys in the store."""
        docs = self.collection.paginated_find()
        for doc in docs:
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
