import asyncio
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class MongodbLoader(BaseLoader):
    """Load MongoDB documents."""
    
    db: Any
    collection: Any

    def __init__(
        self,
        connection_string: str,
        db_name: str,
        collection_name: str,
        filter_criteria: Optional[Dict] = None,
    ):
        if not connection_string:
            raise ValueError("connection_string must be provided.")

        if not db_name:
            raise ValueError("db_name must be provided.")

        if not collection_name:
            raise ValueError("collection_name must be provided.")

        self.client = AsyncIOMotorClient(connection_string)
        self.db_name = db_name
        self.collection_name = collection_name
        self.filter_criteria = filter_criteria or {}

        self.db = self.client.get_database(db_name)
        self.collection = self.db.get_collection(collection_name)

    def load(self) -> List[Document]:
        result = asyncio.run(self._async_load())
        return result

    async def _async_load(self) -> List[Document]:
        result = []

        try:
            cursor = self.collection.find(self.filter_criteria)

            total_docs = self.collection.count_documents(self.filter_criteria)

            async for doc in cursor:
                content = str(doc)
                metadata = {
                    "database": self.db_name,
                    "collection": self.collection_name,
                }
                result.append(Document(page_content=content, metadata=metadata))

            if len(result) != total_docs:
                raise Exception("Only partial collection of documents returned.")

        except Exception as e:
            print(f"Error: {e}")

        return result
