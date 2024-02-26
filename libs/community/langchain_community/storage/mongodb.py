from typing import Iterator, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.stores import BaseStore


class MongoDBStore(BaseStore[str, Document]):
    """BaseStore implementation using MongoDB as the underlying store.

    Examples:
        Create a MongoDBStore instance and perform operations on it:

        .. code-block:: python

            # Instantiate the MongoDBStore with a MongoDB connection
            from langchain.storage import MongoDBStore

            mongo_conn_str = "mongodb://localhost:27017/"
            mongodb_store = MongoDBStore(mongo_conn_str, db_name="test-db",
                                         collection_name="test-collection")

            # Set values for keys
            doc1 = Document(...)
            doc2 = Document(...)
            mongodb_store.mset([("key1", doc1), ("key2", doc2)])

            # Get values for keys
            values = mongodb_store.mget(["key1", "key2"])
            # [doc1, doc2]

            # Iterate over keys
            for key in mongodb_store.yield_keys():
                print(key)

            # Delete keys
            mongodb_store.mdelete(["key1", "key2"])
    """

    def __init__(
        self,
        connection_string: str,
        db_name: str,
        collection_name: str,
        *,
        client_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize the MongoDBStore with a MongoDB connection string.

        Args:
            connection_string (str): MongoDB connection string
            db_name (str): name to use
            collection_name (str): collection name to use
            client_kwargs (dict): Keyword arguments to pass to the Mongo client
        """
        try:
            from pymongo import MongoClient
        except ImportError as e:
            raise ImportError(
                "The MongoDBStore requires the pymongo library to be "
                "installed. "
                "pip install pymongo"
            ) from e

        if not connection_string:
            raise ValueError("connection_string must be provided.")
        if not db_name:
            raise ValueError("db_name must be provided.")
        if not collection_name:
            raise ValueError("collection_name must be provided.")

        self.client = MongoClient(connection_string, **(client_kwargs or {}))
        self.collection = self.client[db_name][collection_name]

    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        """Get the list of documents associated with the given keys.

        Args:
            keys (list[str]): A list of keys representing Document IDs..

        Returns:
            list[Document]: A list of Documents corresponding to the provided
                keys, where each Document is either retrieved successfully or
                represented as None if not found.
        """
        result = self.collection.find({"_id": {"$in": keys}})
        result_dict = {doc["_id"]: Document(**doc["value"]) for doc in result}
        return [result_dict.get(key) for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        """Set the given key-value pairs.

        Args:
            key_value_pairs (list[tuple[str, Document]]): A list of id-document
                pairs.
        Returns:
            None
        """
        from pymongo import UpdateOne

        updates = [{"_id": k, "value": v.__dict__} for k, v in key_value_pairs]
        self.collection.bulk_write(
            [UpdateOne({"_id": u["_id"]}, {"$set": u}, upsert=True) for u in updates]
        )

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given ids.

        Args:
            keys (list[str]): A list of keys representing Document IDs..
        """
        self.collection.delete_many({"_id": {"$in": keys}})

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Yield keys in the store.

        Args:
            prefix (str): prefix of keys to retrieve.
        """
        if prefix is None:
            for doc in self.collection.find(projection=["_id"]):
                yield doc["_id"]
        else:
            for doc in self.collection.find(
                {"_id": {"$regex": f"^{prefix}"}}, projection=["_id"]
            ):
                yield doc["_id"]
