import asyncio
import logging
from typing import Dict, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class MongodbLoader(BaseLoader):
    """Load MongoDB documents."""

    def __init__(
        self,
        connection_string: str,
        db_name: str,
        collection_name: str,
        *,
        filter_criteria: Optional[Dict] = None,
    ) -> None:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError as e:
            raise ImportError(
                "Cannot import from motor, please install with `pip install motor`."
            ) from e
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
        """Load data into Document objects.

        Attention:

        This implementation starts an asyncio event loop which
        will only work if running in a sync env. In an async env, it should
        fail since there is already an event loop running.

        This code should be updated to kick off the event loop from a separate
        thread if running within an async context.
        """
        return asyncio.run(self.aload())

    async def aload(self) -> List[Document]:
        """Load data into Document objects."""
        result = []
        total_docs = await self.collection.count_documents(self.filter_criteria)
        async for doc in self.collection.find(self.filter_criteria):
            metadata = {
                "database": self.db_name,
                "collection": self.collection_name,
            }
            result.append(Document(page_content=str(doc), metadata=metadata))

        if len(result) != total_docs:
            logger.warning(
                f"Only partial collection of documents returned. Loaded {len(result)} "
                f"docs, expected {total_docs}."
            )

        return result
