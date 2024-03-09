from typing import Any, Dict, List, Optional, Sequence

from langchain_community.indexes.base import RecordManager

IMPORT_PYMONGO_ERROR = (
    "Could not import MongoClient. Please install it with `pip install pymongo`."
)
IMPORT_MOTOR_ASYNCIO_ERROR = (
    "Could not import AsyncIOMotorClient. Please install it with `pip install motor`."
)


def _import_pymongo() -> Any:
    """Import PyMongo if available, otherwise raise error."""
    try:
        from pymongo import MongoClient
    except ImportError:
        raise ImportError(IMPORT_PYMONGO_ERROR)
    return MongoClient


def _get_pymongo_client(mongodb_url: str, **kwargs: Any) -> Any:
    """Get MongoClient for sync operations from the mongodb_url,
    otherwise raise error."""
    try:
        pymongo = _import_pymongo()
        client = pymongo(mongodb_url, **kwargs)
    except ValueError as e:
        raise ImportError(
            f"MongoClient string provided is not in proper format. " f"Got error: {e} "
        )
    return client


def _import_motor_asyncio() -> Any:
    """Import Motor if available, otherwise raise error."""
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
    except ImportError:
        raise ImportError(IMPORT_MOTOR_ASYNCIO_ERROR)
    return AsyncIOMotorClient


def _get_motor_client(mongodb_url: str, **kwargs: Any) -> Any:
    """Get AsyncIOMotorClient for async operations from the mongodb_url,
    otherwise raise error."""
    try:
        motor = _import_motor_asyncio()
        client = motor(mongodb_url, **kwargs)
    except ValueError as e:
        raise ImportError(
            f"AsyncIOMotorClient string provided is not in proper format. "
            f"Got error: {e} "
        )
    return client


class MongoDocumentManager(RecordManager):
    """A MongoDB based implementation of the document manager."""

    def __init__(
        self,
        namespace: str,
        *,
        mongodb_url: str,
        db_name: str,
        collection_name: str = "documentMetadata",
    ) -> None:
        """Initialize the MongoDocumentManager.

        Args:
            namespace: The namespace associated with this document manager.
            db_name: The name of the database to use.
            collection_name: The name of the collection to use.
                Default is 'documentMetadata'.
        """
        super().__init__(namespace=namespace)
        self.sync_client = _get_pymongo_client(mongodb_url)
        self.sync_db = self.sync_client[db_name]
        self.sync_collection = self.sync_db[collection_name]
        self.async_client = _get_motor_client(mongodb_url)
        self.async_db = self.async_client[db_name]
        self.async_collection = self.async_db[collection_name]

    def create_schema(self) -> None:
        """Create the database schema for the document manager."""
        pass

    async def acreate_schema(self) -> None:
        """Create the database schema for the document manager."""
        pass

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        """Upsert documents into the MongoDB collection."""
        if group_ids is None:
            group_ids = [None] * len(keys)

        if len(keys) != len(group_ids):
            raise ValueError("Number of keys does not match number of group_ids")

        for key, group_id in zip(keys, group_ids):
            self.sync_collection.find_one_and_update(
                {"namespace": self.namespace, "key": key},
                {"$set": {"group_id": group_id, "updated_at": self.get_time()}},
                upsert=True,
            )

    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        """Asynchronously upsert documents into the MongoDB collection."""
        if group_ids is None:
            group_ids = [None] * len(keys)

        if len(keys) != len(group_ids):
            raise ValueError("Number of keys does not match number of group_ids")

        update_time = await self.aget_time()
        if time_at_least and update_time < time_at_least:
            raise ValueError("Server time is behind the expected time_at_least")

        for key, group_id in zip(keys, group_ids):
            await self.async_collection.find_one_and_update(
                {"namespace": self.namespace, "key": key},
                {"$set": {"group_id": group_id, "updated_at": update_time}},
                upsert=True,
            )

    def get_time(self) -> float:
        """Get the current server time as a timestamp."""
        server_info = self.sync_db.command("hostInfo")
        local_time = server_info["system"]["currentTime"]
        timestamp = local_time.timestamp()
        return timestamp

    async def aget_time(self) -> float:
        """Asynchronously get the current server time as a timestamp."""
        host_info = await self.async_collection.database.command("hostInfo")
        local_time = host_info["system"]["currentTime"]
        return local_time.timestamp()

    def exists(self, keys: Sequence[str]) -> List[bool]:
        """Check if the given keys exist in the MongoDB collection."""
        existing_keys = {
            doc["key"]
            for doc in self.sync_collection.find(
                {"namespace": self.namespace, "key": {"$in": keys}}, {"key": 1}
            )
        }
        return [key in existing_keys for key in keys]

    async def aexists(self, keys: Sequence[str]) -> List[bool]:
        """Asynchronously check if the given keys exist in the MongoDB collection."""
        cursor = self.async_collection.find(
            {"namespace": self.namespace, "key": {"$in": keys}}, {"key": 1}
        )
        existing_keys = {doc["key"] async for doc in cursor}
        return [key in existing_keys for key in keys]

    def list_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """List documents in the MongoDB collection based on the provided date range."""
        query: Dict[str, Any] = {"namespace": self.namespace}
        if before:
            query["updated_at"] = {"$lt": before}
        if after:
            query["updated_at"] = {"$gt": after}
        if group_ids:
            query["group_id"] = {"$in": group_ids}

        cursor = (
            self.sync_collection.find(query, {"key": 1}).limit(limit)
            if limit
            else self.sync_collection.find(query, {"key": 1})
        )
        return [doc["key"] for doc in cursor]

    async def alist_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        Asynchronously list documents in the MongoDB collection
        based on the provided date range.
        """
        query: Dict[str, Any] = {"namespace": self.namespace}
        if before:
            query["updated_at"] = {"$lt": before}
        if after:
            query["updated_at"] = {"$gt": after}
        if group_ids:
            query["group_id"] = {"$in": group_ids}

        cursor = (
            self.async_collection.find(query, {"key": 1}).limit(limit)
            if limit
            else self.async_collection.find(query, {"key": 1})
        )
        return [doc["key"] async for doc in cursor]

    def delete_keys(self, keys: Sequence[str]) -> None:
        """Delete documents from the MongoDB collection."""
        self.sync_collection.delete_many(
            {"namespace": self.namespace, "key": {"$in": keys}}
        )

    async def adelete_keys(self, keys: Sequence[str]) -> None:
        """Asynchronously delete documents from the MongoDB collection."""
        await self.async_collection.delete_many(
            {"namespace": self.namespace, "key": {"$in": keys}}
        )
