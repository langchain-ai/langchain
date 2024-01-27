import base64
from abc import ABC, abstractmethod
from typing import (
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

V = TypeVar("V")


class AstraDBBaseStore(Generic[V], BaseStore[str, V], ABC):
    def __init__(
        self,
        collection_name: str,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_client: Optional[Any] = None,  # 'astrapy.db.AstraDB' if passed
        namespace: Optional[str] = None,
    ) -> None:
        try:
            from astrapy.db import AstraDB, AstraDBCollection
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import a recent astrapy python package. "
                "Please install it with `pip install --upgrade astrapy`."
            )

        # Conflicting-arg checks:
        if astra_db_client is not None:
            if token is not None or api_endpoint is not None:
                raise ValueError(
                    "You cannot pass 'astra_db_client' to AstraDB if passing "
                    "'token' and 'api_endpoint'."
                )

        astra_db = astra_db_client or AstraDB(
            token=token,
            api_endpoint=api_endpoint,
            namespace=namespace,
        )
        self.collection = AstraDBCollection(collection_name, astra_db=astra_db)

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
