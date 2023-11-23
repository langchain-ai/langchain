from mongoengine import Document, connect, get_db
from pymongo import MongoClient
from typing import Iterable, List, Optional
from pprint import pprint
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
        return result

    def run(self, command: str) -> str:
        """Run a command and return a string representing the results."""
        return f"Result:\n{self._execute(command)}"
