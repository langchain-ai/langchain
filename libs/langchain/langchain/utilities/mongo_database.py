from mongoengine.connection import connect, get_db
from typing import Iterable
import pprint

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
    def __init__(self, db_name: str, host: str = 'localhost', port: int = 27017):
        self._db_name = db_name
        self._host = host
        self._port = port

        # Connect to MongoDB using mongoengine
        connect(db=db_name, host=host, port=port)

        self._collections = self._get_available_collections()
        
    @property
    def get_usable_collection_names(self) -> Iterable[str]:
        """Get names of collections available. """
        
        from mongoengine.connection import _get_db
        db = _get_db()
        return db.list_collection_names()
    
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


