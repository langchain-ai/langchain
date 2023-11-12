from mongoengine import connect
from typing import Iterable

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
    
    



