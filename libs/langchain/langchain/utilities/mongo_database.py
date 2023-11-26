"""MongoEngine wrapper around a database."""
from __future__ import annotations

import ast
from pprint import pformat
from typing import Any, Iterable, List, Optional

from pymongo import MongoClient
from pymongo.errors import PyMongoError


def _format_index(index: dict) -> str:
    """Format an index for display."""
    index_keys = index["key"]
    index_keys_formatted = ", ".join(f"{k[0]}: {k[1]}" for k in index_keys)
    unique = ""
    if index_keys[0][0] == "_id" and not index["unique"]:
        unique = ""
    else:
        unique = f' Unique: {index["unique"]},'
    return f'Name: {index["name"]},{unique}' f' Keys: {{ {index_keys_formatted} }}'


class MongoDBDatabase:
    """MongoEngine wrapper around a database."""

    def __init__(
        self,
        client: MongoClient,
        ignore_collections: Optional[List[str]] = None,
        include_collections: Optional[List[str]] = None,
        sample_documents_in_collection_info: int = 3,
    ):
        # Connect to MongoDB using mongoengine
        self._client = client

        if not isinstance(sample_documents_in_collection_info, int):
            raise TypeError("sample_documents_in_collection_info must be an integer")

        db = self._client.get_default_database()
        self._all_collections = set(db.list_collection_names())

        self._include_collections = (
            set(include_collections) if include_collections else set()
        )
        if self._include_collections:
            missing_collections = self._include_collections - self._all_collections
            if missing_collections:
                raise ValueError(
                    f"collections {missing_collections} not found in database"
                )
        self._ignore_collections = (
            set(ignore_collections) if ignore_collections else set()
        )
        if self._ignore_collections:
            missing_collections = self._ignore_collections - self._all_collections
            if missing_collections:
                raise ValueError(
                    f"collections {missing_collections} not found in database"
                )

        if not isinstance(sample_documents_in_collection_info, int):
            raise TypeError("sample_documents_in_collection_info must be an integer")
        self._sample_documents_in_collection_info = sample_documents_in_collection_info

    @classmethod
    def from_uri(cls, database_uri: str, **kwargs: Any) -> MongoDBDatabase:
        """Construct a MongoEngine engine from URI."""
        connection = MongoClient(host=database_uri, **kwargs)
        return cls(connection, **kwargs)

    @property
    def get_usable_collection_names(self) -> Iterable[str]:
        """Get names of collections available."""

        if self._include_collections:
            return sorted(self._include_collections)
        return sorted(self._all_collections - self._ignore_collections)

    def get_document_ids(self, collection_name: str) -> Iterable[str]:
        """Get names of documents available in a given collection."""
        if collection_name not in self._ignore_collections:
            db = self._client.get_default_database()
            # Check if the collection is included or not,
            # if included fetch document ids
            if collection_name in self._include_collections:
                documents = db[collection_name].find()
                return sorted(doc["_id"] for doc in documents)
            else:
                # Fetch all documents in the collection
                documents = db[collection_name].find()
                return sorted(doc["_id"] for doc in documents)
        return []

    @property
    def collection_info(self) -> str:
        """Information about all collections in the database."""
        return self.get_collection_info()

    def get_collection_info(self, collection_names: Optional[List[str]] = None) -> str:
        """Get information about specified collections."""
        all_collection_names = self.get_usable_collection_names
        if collection_names is not None:
            missing_collections = set(collection_names).difference(all_collection_names)
            if missing_collections:
                raise ValueError(
                    f"collection_names {missing_collections} not found in database"
                )
            all_collection_names = collection_names

        collections = []
        for collection_name in all_collection_names:
            db = self._client.get_default_database()
            # Fetch the documents in the collection
            documents = db[collection_name].find()

            # Add document information
            document_info = f"Collection Name: {collection_name}\n"

            # Sample rows or documents info (if required)
            if self._sample_documents_in_collection_info:
                document_info += f"\n{self._get_sample_documents(collection_name)}\n"

            collections.append(document_info)

        collections.sort()
        final_str = "\n\n".join(collections)
        return final_str

    def get_collection_info_no_throw(
        self, collection_names: Optional[List[str]] = None
    ) -> str:
        """Get information about specified collections.

        If the collection does not exist, an error message is returned."""
        try:
            return self.get_collection_info(collection_names)
        except ValueError as e:
            return f"Error: {e}"

    def _get_collection_indexes(self, collection_name: str) -> str:
        """Get indexes of a collection."""
        db = self._client.get_default_database()
        indexes = db[collection_name].index_information()
        indexes_cleaned = [
            {"name": k, "key": v["key"], "unique": "unique" in v and v["unique"]}
            for k, v in indexes.items()
        ]
        indexes_formatted = "\n".join(map(_format_index, indexes_cleaned))
        return f"Collection Indexes:\n{indexes_formatted}"

    def _get_sample_documents(self, collection_name: str) -> str:
        db = self._client.get_default_database()
        documents = (
            db[collection_name].find().limit(self._sample_documents_in_collection_info)
        )
        documents_formatted = "\n".join(map(pformat, documents))
        return (
            f"{self._sample_documents_in_collection_info} sample documents from "
            f"{collection_name}:\n{documents_formatted}"
        )

    def _execute(self, command: str) -> dict[str, Any]:
        """Execute a command and return the result."""
        db = self._client.get_default_database()
        result = db.command(ast.literal_eval(command))
        return pformat(result)

    def run(self, command: str) -> str:
        """Run a command and return a string representing the results."""
        return f"Result:\n{self._execute(command)}"

    def run_no_throw(self, command: str) -> str:
        """Run a command and return a string representing the results.

        If the statement throws an error, the error message is returned."""
        try:
            return self.run(command)
        except PyMongoError as e:
            return f"Error: {e}"
