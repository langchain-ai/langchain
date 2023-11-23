from mongoengine.connection import connect, get_db
from pymongo import MongoClient
from typing import Iterable, List, Optional
import pprint
import ast

def _format_index(index: dict) -> str:
    """Format an index for display."""
    index_keys = index["key"]
    index_keys_formatted = ", ".join(f"{k[0]}: {k[1]}" for k in index_keys)
    return (
        f'Name: {index["name"]}, Unique: {index["unique"]},'
        f' Keys: {{ {index_keys_formatted} }}'
    )


class MongoDBDatabase:
    """MongoEngine wrapper around a database."""

    def __init__(
            self,
            client: MongoClient,
            ignore_collections: Optional[List[str]] = None,
            include_collections: Optional[List[str]] = None,
            sample_documents_in_collection_info: int = 3
    ):

        # Connect to MongoDB using mongoengine
        self._client = client

        if not isinstance(sample_documents_in_collection_info, int):
            raise TypeError("sample_documents_in_collection_info must be an integer")
        
        self._all_collections = set(get_db().list_collection_names())

        self._include_collections = set(include_collections) if include_collections else set()
        if self._include_collections:
            missing_collections = self._include_collections - self._all_collections
            if missing_collections:
                raise ValueError(
                    f"collections {missing_collections} not found in database"
                )
        self._ignore_collections = set(ignore_collections) if ignore_collections else set()
        if self._ignore_collections:
            missing_collections = self._ignore_collections - self._all_collections
            if missing_collections:
                raise ValueError(
                    f"collections {missing_collections} not found in database"
                )

        if not isinstance(sample_documents_in_collection_info, int):
            raise TypeError("sample_documents_in_collection_info must be an integer")
        self._sample_rows_in_table_info = sample_documents_in_collection_info

    @classmethod
    def from_uri(cls, database_uri: str, **kwargs):
        """Construct a MongoEngine engine from URI."""
        connection = connect(host=database_uri, **kwargs)
        return cls(connection, **kwargs)

    @property
    def get_usable_collection_names(self) -> Iterable[str]:
        """Get names of collections available. """
        
        if self._include_collections:
            return sorted(self._include_collections)
        return sorted(self._all_collections - self._ignore_collections)
    
    @property
    def document_info(self, collection_name: str):
        """Information about all documents in the database."""
        collection = eval(collection_name)
        pprint(collection._fields)
        
    def get_collection_info(self, collection_name: str) -> str:
        """Information about a specific collection"""
        collection = eval(collection_name)
        fields_info = collection._fields

        formatted_info = pprint.pformat(fields_info)

        return f"Collection Information for '{collection_name}':\n{formatted_info}"
    
    def _get_collection_indexes(self, collection_name: str) -> str:
        """Get indexes of a collection."""
        db = get_db()
        indexes = db[collection_name].index_information()
        indexes_cleaned = [{"name": k, "key": v["key"], "unique": "unique" in v} 
                           for k, v in indexes.items()]
        indexes_formatted = "\n".join(map(_format_index, indexes_cleaned))
        return f"Collection Indexes:\n{indexes_formatted}"

    def _get_sample_documents(self, collection_name: str) -> str:
        db = get_db()
        documents = db[collection_name].find().limit(
            self._sample_documents_in_collection_info
        )
        documents_formatted = pprint.pformat(list(documents))
        return f"Sample Documents:\n{documents_formatted}"
    
    def _execute(self, command: str) -> str:
        """Execute a command and return the result."""
        db = get_db()
        result = db.command(ast.literal_eval(command))
        return f"Result:\n{result}"

