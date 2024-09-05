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
        metadata_mapper: Optional[Callable[..., Dict[str, Any]]] = None,
        enable_total_count_check: bool = True,
    ) -> None:
        """
        Args:
            connection_string: The connection string for the MongoDB database.
            db_name: The name of the MongoDB database.
            collection_name: The name of the collection within the database.
            filter_criteria: Optional dictionary to filter documents in the
                collection. Defaults to an empty dictionary if not provided.
            field_names: Optional sequence of field names to include in the
                loaded documents. If not provided, all fields will be included.
            cursor_builder: Optional function to build a cursor for querying
                the collection. Defaults to `default_cursor_builder` if not
                provided.
            page_content_mapper: Optional function to map a document to its
                page content. Defaults to `page_content_default_mapper` if
                not provided.
            metadata_mapper: Optional function to map a document to its metadata.
                Defaults to `metadata_default_mapper` if not provided.
            enable_total_count_check: Whether to check the total count of
                documents in the collection to ensure completeness. Defaults
                to True.
        Raises:
            ValueError: If `connection_string`, `db_name`, or `collection_name`
                is not provided.
            ImportError: If `motor` is not installed and cannot be imported.
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
        self.field_names = field_names
        self.filter_criteria = filter_criteria or {}

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
        return {
            "database": db_name,
            "collection": collection_name,
        }

    @staticmethod
    def page_content_default_mapper(
        doc: Dict, field_names: Optional[Sequence[str]] = None
    ) -> str:
        """
        Converts a record into a "page content" string.
        """
        # Extract text content from filtered fields or use the entire document
        if field_names is not None:
            fields = {}
            for name in field_names:
                # Split the field names to handle nested fields
                keys = name.split(".")
                value: Union[Dict, str] = doc
                for key in keys:
                    if key in doc:
                        value = doc[key]
                    else:
                        value = ""
                        break
                fields[name] = value

            texts = [str(value) for value in fields.values()]
            text = " ".join(texts)
        else:
            text = str(doc)
        return text

    async def aload(self) -> List[Document]:
        """Load data into Document objects."""
        result = []
        if self.enable_total_count_check:
            total_docs = await self.collection.count_documents(self.filter_criteria)

        # Construct the projection dictionary if field_names are specified
        projection = (
            {field: 1 for field in self.field_names} if self.field_names else None
        )

        async for doc in self.cursor_builder(
            self.collection, self.filter_criteria, projection
        ):
            metadata = self.metadata_mapper(self.db_name, self.collection_name, doc)
            page_content = self.page_content_mapper(doc, self.field_names)
            result.append(Document(page_content=page_content, metadata=metadata))

        if self.enable_total_count_check and len(result) != total_docs:
            logger.warning(
                f"Only partial collection of documents returned. "
                f"Loaded {len(result)} docs, expected {total_docs}."
            )
        return result
