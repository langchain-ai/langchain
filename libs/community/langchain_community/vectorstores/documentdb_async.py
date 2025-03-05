from __future__ import annotations

import logging
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from motor.core import AgnosticClient, AgnosticCollection
    from pymongo.collection import Collection


# Before Python 3.11 native StrEnum is not available
class DocumentDBSimilarityType(str, Enum):
    """DocumentDB Similarity Type as enumerator."""

    COS = "cosine"
    """Cosine similarity"""
    DOT = "dotProduct"
    """Dot product"""
    EUC = "euclidean"
    """Euclidean distance"""


DocumentDBDocumentType = TypeVar("DocumentDBDocumentType", bound=Dict[str, Any])

logger = logging.getLogger(__name__)

DEFAULT_INSERT_BATCH_SIZE = 128


class DocumentDBVectorSearch(VectorStore):
    """`Amazon DocumentDB (with MongoDB compatibility)` vector store.
    Please refer to the official Vector Search documentation for more details:
    https://docs.aws.amazon.com/documentdb/latest/developerguide/vector-search.html

    To use, you should have both:
    - the ``pymongo`` python package installed
    - a connection string and credentials associated with a DocumentDB cluster

    Example:
        . code-block:: python

            from langchain_aws.vectorstores import DocumentDBVectorSearch
            from langchain_aws.embeddings.openai import OpenAIEmbeddings
            from pymongo import MongoClient

            mongo_client = MongoClient("<YOUR-CONNECTION-STRING>")
            collection = mongo_client["<db_name>"]["<collection_name>"]
            embeddings = OpenAIEmbeddings()
            vectorstore = DocumentDBVectorSearch(collection, embeddings)
    """

    def __init__(
        self,
        collection: Collection[DocumentDBDocumentType],
        embedding: Embeddings,
        *,
        index_name: str = "vectorSearchIndex",
        text_key: str = "textContent",
        embedding_key: str = "vectorContent",
        is_async: bool = False,
        async_collection: AgnosticCollection,
    ):
        """Constructor for DocumentDBVectorSearch

        Args:
            collection: MongoDB collection to add the texts to.
            embedding: Text embedding model to use.
            index_name: Name of the Vector Search index.
            text_key: MongoDB field that will contain the text
                for each document.
            embedding_key: MongoDB field that will contain the embedding
                for each document.
            is_async: Whether the collection is async or not.
            async_collection: Async version of the collection.
            NOTES:
                * If `is_async` is True, `async_collection` must be provided.
                * `collection` must be provided also when `is_async` is False.
        """
        if collection is None:
            raise ValueError("Must provide 'collection' named parameter.")
        self._collection = collection
        self._embedding = embedding
        self._index_name = index_name
        self._text_key = text_key
        self._embedding_key = embedding_key
        self._similarity_type = DocumentDBSimilarityType.COS
        self.is_async = is_async
        if is_async and async_collection is None:
            raise ValueError(
                f"Expecting `async_collection` when `is_async` is defined.\n \
                    Go async_collection = `{async_collection}`"
            )
        self._async_collection = async_collection  # type: ignore[arg-type]

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def validate_async(self) -> None:
        if not self.is_async:
            raise RuntimeError(
                f"Async functions can only be called \n \
                    when the object `is_async` flag is `True`.\n \
                               is_async = `{self.is_async}`"
            )

    def get_index_name(self) -> str:
        """Returns the index name

        Returns:
            Returns the index name

        """
        return self._index_name

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        namespace: str,
        embedding: Embeddings,
        **kwargs: Any,
    ) -> DocumentDBVectorSearch:
        """Creates an Instance of DocumentDBVectorSearch from a Connection String

        Args:
            connection_string: The DocumentDB cluster endpoint connection string
            namespace: The namespace (database.collection)
            embedding: The embedding utility
            **kwargs: Dynamic keyword arguments

        Returns:
            an instance of the vector store

        """
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(
                "Could not import pymongo, please install it with "
                "`pip install pymongo`."
            )
        client: MongoClient = MongoClient(connection_string)
        db_name, collection_name = namespace.split(".")
        collection = client[db_name][collection_name]
        return cls(collection, embedding, **kwargs)

    @classmethod
    def afrom_connection_string(
        cls,
        connection_string: str,
        namespace: str,
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "DocumentDBVectorSearch":
        """Creates an Instance of DocumentDBVectorSearch from a Connection
        String.

        Args:
            connection_string: The DocumentDB cluster endpoint connection string
            namespace: The namespace (database.collection)
            embedding: The embedding utility
            **kwargs: Dynamic keyword arguments

        Returns:
            an instance of the vector store
        """
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(
                "Could not import pymongo, please install it with "
                "`pip install pymongo`."
            )
        client: MongoClient = MongoClient(connection_string)
        try:
            from motor.core import AgnosticClient  # noqa: F811 F401
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            raise ImportError(
                "Could not import motor, please install it with `pip install motor`."
            )
        async_client: AgnosticClient = AsyncIOMotorClient(connection_string)
        db_name, collection_name = namespace.split(".")
        collection = client[db_name][collection_name]
        async_collection = async_client[db_name][collection_name]
        return cls(
            collection,
            embedding,
            is_async=True,
            async_collection=async_collection,
            **kwargs,
        )

    def index_exists(self) -> bool:
        """Verifies if the specified index name during instance
            construction exists on the collection

        Returns:
          Returns True on success and False if no such index exists
            on the collection
        """
        cursor = self._collection.list_indexes()
        index_name = self._index_name

        for res in cursor:
            current_index_name = res.pop("name")
            if current_index_name == index_name:
                return True

        return False

    async def aindex_exists(self) -> bool:
        """Verifies if the specified index name during instance construction
        exists on the collection.

        Returns:
          Returns True on success and False if no such index exists
            on the collection
        """
        self.validate_async()
        cursor = self._async_collection.list_indexes()
        index_name = self._index_name

        async for res in cursor:
            current_index_name = res.pop("name")
            if current_index_name == index_name:
                return True

        return False

    def delete_index(self) -> None:
        """Deletes the index specified during instance construction if it exists"""
        if self.index_exists():
            self._collection.drop_index(self._index_name)
            # Raises OperationFailure on an error (e.g. trying to drop
            # an index that does not exist)

    async def adelete_index(self) -> None:
        """Deletes the index specified during instance construction if it
        exists."""
        self.validate_async()
        if await self.aindex_exists():
            await self._async_collection.drop_index(self._index_name)
            # Raises OperationFailure on an error (e.g. trying to drop
            # an index that does not exist)

    def create_index(
        self,
        dimensions: int = 1536,
        similarity: DocumentDBSimilarityType = DocumentDBSimilarityType.COS,
        m: int = 16,
        ef_construction: int = 64,
    ) -> dict[str, Any]:
        """Creates an index using the index name specified at
            instance construction

        Args:
            dimensions: Number of dimensions for vector similarity.
                The maximum number of supported dimensions is 2000

            similarity: Similarity algorithm to use with the HNSW index.

            m: Specifies the max number of connections for an HNSW index.
                Large impact on memory consumption.

            ef_construction: Specifies the size of the dynamic candidate list
                for constructing the graph for HNSW index. Higher values lead
                to more accurate results but slower indexing speed.

                Possible options are:
                    - DocumentDBSimilarityType.COS (cosine distance),
                    - DocumentDBSimilarityType.EUC (Euclidean distance), and
                    - DocumentDBSimilarityType.DOT (dot product).

        Returns:
            An object describing the created index

        """
        self._similarity_type = similarity

        # prepare the command
        create_index_commands = {
            "createIndexes": self._collection.name,
            "indexes": [
                {
                    "name": self._index_name,
                    "key": {self._embedding_key: "vector"},
                    "vectorOptions": {
                        "type": "hnsw",
                        "similarity": similarity,
                        "dimensions": dimensions,
                        "m": m,
                        "efConstruction": ef_construction,
                    },
                }
            ],
        }

        # retrieve the database object
        current_database = self._collection.database

        # invoke the command from the database object
        create_index_responses: dict[str, Any] = current_database.command(
            create_index_commands
        )

        return create_index_responses

    async def acreate_index(
        self,
        dimensions: int = 1536,
        similarity: DocumentDBSimilarityType = DocumentDBSimilarityType.COS,
        m: int = 16,
        ef_construction: int = 64,
    ) -> dict[str, Any]:
        """Creates an index using the index name specified at instance
        construction.

        Args:
            dimensions: Number of dimensions for vector similarity.
                The maximum number of supported dimensions is 2000

            similarity: Similarity algorithm to use with the HNSW index.

            m: Specifies the max number of connections for an HNSW index.
                Large impact on memory consumption.

            ef_construction: Specifies the size of the dynamic candidate list
                for constructing the graph for HNSW index. Higher values lead
                to more accurate results but slower indexing speed.

                Possible options are:
                    - DocumentDBSimilarityType.COS (cosine distance),
                    - DocumentDBSimilarityType.EUC (Euclidean distance), and
                    - DocumentDBSimilarityType.DOT (dot product).

        Returns:
            An object describing the created index
        """
        self.validate_async()
        self._similarity_type = similarity

        # prepare the command
        create_index_commands = {
            "createIndexes": self._async_collection.name,
            "indexes": [
                {
                    "name": self._index_name,
                    "key": {self._embedding_key: "vector"},
                    "vectorOptions": {
                        "type": "hnsw",
                        "similarity": similarity,
                        "dimensions": dimensions,
                        "m": m,
                        "efConstruction": ef_construction,
                    },
                }
            ],
        }

        # retrieve the database object
        current_database = self._async_collection.database

        # invoke the command from the database object
        create_index_responses: dict[str, Any] = await current_database.command(
            create_index_commands
        )

        return create_index_responses

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List:
        batch_size = kwargs.get("batch_size", DEFAULT_INSERT_BATCH_SIZE)
        _metadatas: Union[List, Generator] = metadatas or ({} for _ in texts)
        texts_batch = []
        metadatas_batch = []
        result_ids = []
        for i, (text, metadata) in enumerate(zip(texts, _metadatas)):
            texts_batch.append(text)
            metadatas_batch.append(metadata)
            if (i + 1) % batch_size == 0:
                result_ids.extend(self._insert_texts(texts_batch, metadatas_batch))
                texts_batch = []
                metadatas_batch = []
        if texts_batch:
            result_ids.extend(self._insert_texts(texts_batch, metadatas_batch))
        return result_ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List:
        self.validate_async()
        batch_size = kwargs.get("batch_size", DEFAULT_INSERT_BATCH_SIZE)
        _metadatas: Union[List, Generator] = metadatas or ({} for _ in texts)
        texts_batch = []
        metadatas_batch = []
        result_ids = []
        for i, (text, metadata) in enumerate(zip(texts, _metadatas)):
            texts_batch.append(text)
            metadatas_batch.append(metadata)
            if (i + 1) % batch_size == 0:
                new_result_ids = await self._ainsert_texts(texts_batch, metadatas_batch)
                result_ids.extend(new_result_ids)
                texts_batch = []
                metadatas_batch = []
        if texts_batch:
            new_result_ids = await self._ainsert_texts(texts_batch, metadatas_batch)
            result_ids.extend(new_result_ids)
        return result_ids

    async def _ainsert_texts(
        self, texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> List:
        """Used to Load Documents into the collection.

        Args:
            texts: The list of documents strings to load
            metadatas: The list of metadata objects associated with each document

        Returns:
        """
        self.validate_async()
        # If the text is empty, then exit early
        if not texts:
            return []

        # Embed and create the documents
        embeddings = self._embedding.embed_documents(texts)
        to_insert = [
            {self._text_key: t, self._embedding_key: embedding, **m}
            for t, m, embedding in zip(texts, metadatas, embeddings)
        ]
        # insert the documents in DocumentDB
        insert_result = await self._async_collection.insert_many(to_insert)  # type: ignore
        return insert_result.inserted_ids

    def _insert_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List:
        """Used to Load Documents into the collection

        Args:
            texts: The list of documents strings to load
            metadatas: The list of metadata objects associated with each document

        Returns:

        """
        # If the text is empty, then exit early
        if not texts:
            return []

        # Embed and create the documents
        embeddings = self._embedding.embed_documents(texts)
        to_insert = [
            {self._text_key: t, self._embedding_key: embedding, **m}
            for t, m, embedding in zip(texts, metadatas, embeddings)
        ]
        # insert the documents in DocumentDB
        insert_result = self._collection.insert_many(to_insert)  # type: ignore
        return insert_result.inserted_ids

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection: Optional[Collection[DocumentDBDocumentType]] = None,
        **kwargs: Any,
    ) -> DocumentDBVectorSearch:
        try:
            assert isinstance(collection, Collection)
        except AssertionError:
            raise ValueError("Must provide 'collection' named parameter.")
        vectorstore = cls(collection, embedding, **kwargs)  # type: ignore[arg-type]
        vectorstore.add_texts(texts, metadatas=metadatas)
        return vectorstore

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection: Optional[Collection[DocumentDBDocumentType]] = None,
        async_collection: Optional[AgnosticCollection] = None,
        **kwargs: Any,
    ) -> DocumentDBVectorSearch:
        if collection is None or async_collection is None:
            raise ValueError(
                f"Must provide 'collection' and `async_collection` named parameters.\n \
                    got collection: `{collection}`\n \
                    async_collection: `{async_collection}`"
            )
        vectorstore = cls(
            collection,  # type: ignore [arg-type]
            embedding,
            is_async=True,
            async_collection=async_collection,  # type: ignore [arg-type]
            **kwargs,
        )
        await vectorstore.aadd_texts(texts, metadatas=metadatas)
        return vectorstore

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is None:
            raise ValueError("No document ids provided to delete.")

        for document_id in ids:
            self.delete_document_by_id(document_id)
        return True

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        self.validate_async()
        if ids is None:
            raise ValueError("No document ids provided to delete.")

        for document_id in ids:
            await self.adelete_document_by_id(document_id)
        return True

    def delete_document_by_id(self, document_id: Optional[str] = None) -> None:
        """Removes a Specific Document by Id

        Args:
            document_id: The document identifier
        """
        try:
            from bson.objectid import ObjectId
        except ImportError as e:
            raise ImportError(
                "Unable to import bson, please install with `pip install bson`."
            ) from e
        if document_id is None:
            raise ValueError("No document id provided to delete.")

        self._collection.delete_one({"_id": ObjectId(document_id)})

    async def adelete_document_by_id(self, document_id: Optional[str] = None) -> None:
        """Removes a Specific Document by Id.

        Args:
            document_id: The document identifier
        """
        self.validate_async()
        try:
            from bson.objectid import ObjectId
        except ImportError as e:
            raise ImportError(
                "Unable to import bson, please install with `pip install bson`."
            ) from e
        if document_id is None:
            raise ValueError("No document id provided to delete.")

        await self._async_collection.delete_one({"_id": ObjectId(document_id)})

    def _similarity_search_without_score(
        self,
        embeddings: List[float],
        k: int = 4,
        ef_search: int = 40,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Returns a list of documents.

        Args:
            embeddings: The query vector
            k: the number of documents to return
            ef_search: Specifies the size of the dynamic candidate list
                that HNSW index uses during search. A higher value of
                efSearch provides better recall at cost of speed.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
        Returns:
            A list of documents closest to the query vector
        """
        # $match can't be null, so intializes to {} when None to avoid
        # "the match filter must be an expression in an object"
        if not filter:
            filter = {}
        pipeline: List[dict[str, Any]] = [
            {"$match": filter},
            {
                "$search": {
                    "vectorSearch": {
                        "vector": embeddings,
                        "path": self._embedding_key,
                        "similarity": self._similarity_type,
                        "k": k,
                        "efSearch": ef_search,
                    }
                },
            },
        ]

        cursor = self._collection.aggregate(pipeline)

        docs = []

        for res in cursor:
            text = res.pop(self._text_key)
            docs.append(Document(page_content=text, metadata=res))

        return docs

    async def _asimilarity_search_without_score(
        self,
        embeddings: List[float],
        k: int = 4,
        ef_search: int = 40,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Returns a list of documents.

        Args:
            embeddings: The query vector
            k: the number of documents to return
            ef_search: Specifies the size of the dynamic candidate list
                that HNSW index uses during search. A higher value of
                efSearch provides better recall at cost of speed.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
        Returns:
            A list of documents closest to the query vector
        """
        self.validate_async()
        # $match can't be null, so intializes to {} when None to avoid
        # "the match filter must be an expression in an object"
        if not filter:
            filter = {}
        pipeline: List[dict[str, Any]] = [
            {"$match": filter},
            {
                "$search": {
                    "vectorSearch": {
                        "vector": embeddings,
                        "path": self._embedding_key,
                        "similarity": self._similarity_type,
                        "k": k,
                        "efSearch": ef_search,
                    },
                },
            },
        ]
        cursor = self._async_collection.aggregate(pipeline)

        docs = []

        async for res in cursor:
            text = res.pop(self._text_key)
            docs.append(Document(page_content=text, metadata=res))

        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        ef_search: int = 40,
        *,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embeddings = self._embedding.embed_query(query)
        docs = self._similarity_search_without_score(
            embeddings=embeddings, k=k, ef_search=ef_search, filter=filter
        )
        return [doc for doc in docs]

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        ef_search: int = 40,
        *,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        self.validate_async()
        embeddings = self._embedding.embed_query(query)
        docs = await self._asimilarity_search_without_score(
            embeddings=embeddings, k=k, ef_search=ef_search, filter=filter
        )
        return [doc for doc in docs]
