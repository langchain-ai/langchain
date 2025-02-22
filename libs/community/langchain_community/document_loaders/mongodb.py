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
        metadata_names: Optional[Sequence[str]] = None,
        include_db_collection_in_metadata: bool = True,
    ) -> None:
        """
        Initializes the MongoDB loader with necessary database connection
        details and configurations.

        Args:
            connection_string (str): MongoDB connection URI.
            db_name (str):Name of the database to connect to.
            collection_name (str): Name of the collection to fetch documents from.
            filter_criteria (Optional[Dict]): MongoDB filter criteria for querying
            documents.
            field_names (Optional[Sequence[str]]): List of field names to retrieve
            from documents.
            metadata_names (Optional[Sequence[str]]): Additional metadata fields to
            extract from documents.
            include_db_collection_in_metadata (bool): Flag to include database and
            collection names in metadata.

        Raises:
            ImportError: If the motor library is not installed.
            ValueError: If any necessary argument is missing.
        """
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
        self.field_names = field_names or []
        self.filter_criteria = filter_criteria or {}
        self.metadata_names = metadata_names or []
        self.include_db_collection_in_metadata = include_db_collection_in_metadata

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
        """Asynchronously loads data into Document objects."""
        result = []
        total_docs = await self.collection.count_documents(self.filter_criteria)

        projection = self._construct_projection()

        async for doc in self.collection.find(self.filter_criteria, projection):
            metadata = self._extract_fields(doc, self.metadata_names, default="")

            # Optionally add database and collection names to metadata
            if self.include_db_collection_in_metadata:
                metadata.update(
                    {
                        "database": self.db_name,
                        "collection": self.collection_name,
                    }
                )

            # Extract text content from filtered fields or use the entire document
            if self.field_names is not None:
                fields = self._extract_fields(doc, self.field_names, default="")
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

    def _construct_projection(self) -> Optional[Dict]:
        """Constructs the projection dictionary for MongoDB query based
        on the specified field names and metadata names."""
        field_names = list(self.field_names) or []
        metadata_names = list(self.metadata_names) or []
        all_fields = field_names + metadata_names
        return {field: 1 for field in all_fields} if all_fields else None

    def _extract_fields(
        self,
        document: Dict,
        fields: Sequence[str],
        default: str = "",
    ) -> Dict:
        """Extracts and returns values for specified fields from a document."""
        extracted = {}
        for field in fields or []:
            value = document
            for key in field.split("."):
                value = value.get(key, default)
                if value == default:
                    break
            new_field_name = field.replace(".", "_")
            extracted[new_field_name] = value
        return extracted
