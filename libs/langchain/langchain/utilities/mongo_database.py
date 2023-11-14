from mongoengine import Document, StringField, IntField, ListField, connect
from typing import Iterable, List, Optional
from pprint import pprint

class MongoDBDatabase:
    """MongoEngine wrapper around a database."""
    def __init__(self, db_name: str, host: str = 'localhost', port: int = 27017):
        self._db_name = db_name
        self._host = host
        self._port = port

        # Connect to MongoDB using mongoengine
        connect(db=db_name, host=host, port=port)

        self._collections = self._get_available_collections()
        
    @classmethod
    def from_uri(cls, database_uri: str, **kwargs):
        """Construct a MongoEngine engine from URI."""
        connection = connect(host=database_uri, **kwargs)
        return cls(connection, **kwargs)
    
        
    @property
    def get_usable_collection_names(self) -> Iterable[str]:
        """Get names of collections available. """
        
        from mongoengine.connection import _get_db
        db = _get_db()
        return db.list_collection_names()
    
    
    def get_usable_document_names(self, collection_name: str) -> Iterable[str]:
        """Get names of documents available in a given collection."""
        if collection_name not in self._ignore_collections:
            # Check if the collection is included or not, if included fetch document names
            if collection_name in self._include_collections:
                documents = Document._get_collection(name=collection_name).find()
                return sorted(doc["_id"] for doc in documents)
            else:
                # Fetch all documents in the collection
                documents = Document._get_collection(name=collection_name).find()
                return sorted(doc["_id"] for doc in documents)
        return []
    
    @property
    def document_info(self, collection_name: str):
        """Information about all documents in the database."""
        return self.get_document_info()
        
    def collection_info(self) -> str:
        """Information about all collections in the database."""
        return self.get_collection_info()
        
        
    def get_collection_info(self, collection_name: str) -> str:
        """Information about a specific collection"""
        collection = eval(collection_name)
        fields_info = collection._fields

        formatted_info = pprint.pformat(fields_info)

        return f"Collection Information for '{collection_name}':\n{formatted_info}"
    
    
    def get_document_info(self, collection_names: Optional[List[str]] = None) -> str:
        """Get information about specified collections."""
        all_collection_names = self.get_usable_collection_names()
        if collection_names is not None:
            missing_collections = set(collection_names).difference(all_collection_names)
            if missing_collections:
                raise ValueError(f"collection_names {missing_collections} not found in database")
            all_collection_names = collection_names

        collections = []
        for collection_name in all_collection_names:
            if collection_name in self._custom_document_info:
                collections.append(self._custom_document_info[collection_name])
                continue

            # Fetch the documents in the collection
            documents = Document._get_collection(name=collection_name).find()

            # Add document information
            document_info = f"Collection Name: {collection_name}\n"

            # Sample rows or documents info (if required)
            if self._sample_documents_in_info:
                document_info += "\nSample Documents:\n"
                # Fetch a specified number of sample documents (you can adjust this number)
                sample_documents = [doc for _, doc in zip(range(self._sample_documents_count), documents)]
                for sample_doc in sample_documents:
                    document_info += f"{sample_doc}\n"
                document_info += "\n"

            collections.append(document_info)

        collections.sort()
        final_str = "\n\n".join(collections)
        return final_str
    
    



