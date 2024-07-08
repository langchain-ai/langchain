import asyncio
import logging
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

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
        field_names: Optional[Sequence[str]] = None,
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
        self.field_names = field_names
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

        # Construct the projection dictionary if field_names are specified
        projection = (
            {field: 1 for field in self.field_names} if self.field_names else None
        )

        async for doc in self.collection.find(self.filter_criteria, projection):
            metadata = {
                "database": self.db_name,
                "collection": self.collection_name,
            }

            # Extract text content from filtered fields or use the entire document
            if self.field_names is not None:
                fields = {}
                for name in self.field_names:
                    # Split the field names to handle nested fields
                    keys = name.split(".")
                    value = doc
                    for key in keys:
                        if key in value:
                            value = value[key]
                        else:
                            value = ""
                            break
                    fields[name] = value

                texts = [str(value) for value in fields.values()]
                text = " ".join(texts)
            else:
                text = str(doc)

            result.append(Document(page_content=text, metadata=metadata))

        if len(result) != total_docs:
            logger.warning(
                f"Only partial collection of documents returned. "
                f"Loaded {len(result)} docs, expected {total_docs}."
            )

        return result
