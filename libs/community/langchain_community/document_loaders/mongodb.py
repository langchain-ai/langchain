import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

from langchain_core.documents import Document

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorCursor

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
        cursor_builder: Optional[Callable] = None,
        page_content_mapper: Optional[Callable[..., str]] = None,
        include_db_collection_in_metadata: bool = True,
        metadata_names: Optional[Sequence[str]] = None,
        metadata_mapper: Optional[Callable[..., Dict[str, Any]]] = None,
        enable_total_count_check: bool = True,
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
            cursor_builder (Optional[Callable]): Optional function to build a cursor
            for querying the collection. Defaults to `default_cursor_builder` if
            not provided.
            page_content_mapper (Optional[Callable[..., str]]): Optional function to
            map a document to its page content. Defaults to 
            `page_content_default_mapper` if not provided.
            metadata_names (Optional[Sequence[str]]): Additional metadata fields to
            extract from documents.
            include_db_collection_in_metadata (bool): Flag to include database and
            collection names in metadata.
            metadata_mapper (Optional[Callable[..., Dict[str, Any]]]): Optional 
            function to map a document to its metadata. Defaults to 
            `metadata_default_mapper` if not provided.
            enable_total_count_check (bool): Whether to check the total count of
            documents in the collection to ensure completeness. Defaults to True.

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

        self.client: AsyncIOMotorClient = AsyncIOMotorClient(connection_string)
        self.db_name = db_name
        self.collection_name = collection_name
        self.field_names = field_names or []
        self.filter_criteria = filter_criteria or {}
        self.metadata_names = metadata_names or []
        self.include_db_collection_in_metadata = include_db_collection_in_metadata

        self.db = self.client.get_database(db_name)
        self.collection = self.db.get_collection(collection_name)

        self.cursor_builder = cursor_builder or self.default_cursor_builder
        self.metadata_mapper = metadata_mapper or self.metadata_default_mapper
        self.page_content_mapper = (
            page_content_mapper or self.page_content_default_mapper
        )
        self.enable_total_count_check = enable_total_count_check

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

    @staticmethod
    def default_cursor_builder(
        collection: "AsyncIOMotorCollection",
        filter_criteria: Optional[Dict] = None,
        projection: Optional[Dict[str, Any]] = None,
    ) -> "AsyncIOMotorCursor":
        return collection.find(filter_criteria, projection)

    @staticmethod
    def metadata_default_mapper(
        db_name: str,
        collection_name: str,
        doc: Dict,
    ) -> Dict[str, Any]:
        """
        Converts a doc into a "metadata" dictionary.
        """
        return self._extract_fields(doc, self.metadata_names, default="")

    @staticmethod
    def page_content_default_mapper(
        doc: Dict, field_names: Optional[Sequence[str]] = None
    ) -> str:
        """
        Converts a record into a "page content" string.
        """
        # Extract text content from filtered fields or use the entire document
        if self.field_names is not None:
            fields = self._extract_fields(doc, self.field_names, default="")
            texts = [str(value) for value in fields.values()]
            text = " ".join(texts)
        else:
            text = str(doc)
        return text

    async def aload(self) -> List[Document]:
        """Asynchronously loads data into Document objects."""
        result = []
        if self.enable_total_count_check:
            total_docs = await self.collection.count_documents(self.filter_criteria)

        projection = self._construct_projection()

        async for doc in self.cursor_builder(
            self.collection, self.filter_criteria, projection
        ):
            metadata = self.metadata_mapper(self.db_name, self.collection_name, doc)
           
            # Optionally add database and collection names to metadata
            if self.include_db_collection_in_metadata:
                metadata.update(
                    {"database": self.db_name, "collection": self.collection_name}
                )
            
            page_content = self.page_content_mapper(doc, self.field_names)
            result.append(Document(page_content=page_content, metadata=metadata))

        if self.enable_total_count_check and len(result) != total_docs:
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
