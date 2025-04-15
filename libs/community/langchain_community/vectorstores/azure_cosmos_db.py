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
    Tuple,
    Union,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from pymongo.collection import Collection


# Before Python 3.11 native StrEnum is not available
class CosmosDBSimilarityType(str, Enum):
    """Cosmos DB Similarity Type as enumerator."""

    COS = "COS"
    """CosineSimilarity"""
    IP = "IP"
    """inner - product"""
    L2 = "L2"
    """Euclidean distance"""


class CosmosDBVectorSearchType(str, Enum):
    """Cosmos DB Vector Search Type as enumerator."""

    VECTOR_IVF = "vector-ivf"
    """IVF vector index"""
    VECTOR_HNSW = "vector-hnsw"
    """HNSW vector index"""
    VECTOR_DISKANN = "vector-diskann"
    """DISKANN vector index"""


logger = logging.getLogger(__name__)

DEFAULT_INSERT_BATCH_SIZE = 128


class AzureCosmosDBVectorSearch(VectorStore):
    """`Azure Cosmos DB for MongoDB vCore` vector store.

    To use, you should have both:
    - the ``pymongo`` python package installed
    - a connection string associated with a MongoDB VCore Cluster

    Example:
        . code-block:: python

            from langchain_community.vectorstores import
            AzureCosmosDBVectorSearch
            from langchain_community.embeddings.openai import OpenAIEmbeddings
            from pymongo import MongoClient

            mongo_client = MongoClient("<YOUR-CONNECTION-STRING>")
            collection = mongo_client["<db_name>"]["<collection_name>"]
            embeddings = OpenAIEmbeddings()
            vectorstore = AzureCosmosDBVectorSearch(collection, embeddings)
    """

    def __init__(
        self,
        collection: Collection,
        embedding: Embeddings,
        *,
        index_name: str = "vectorSearchIndex",
        text_key: str = "textContent",
        embedding_key: str = "vectorContent",
        application_name: str = "LangChain-CDBMongoVCore-VectorStore-Python",
    ):
        """Constructor for AzureCosmosDBVectorSearch

        Args:
            collection: MongoDB collection to add the texts to.
            embedding: Text embedding model to use.
            index_name: Name of the Atlas Search index.
            text_key: MongoDB field that will contain the text
                for each document.
            embedding_key: MongoDB field that will contain the embedding
                for each document.
        """
        self._collection = collection
        self._embedding = embedding
        self._index_name = index_name
        self._text_key = text_key
        self._embedding_key = embedding_key
        self._application_name = application_name

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

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
        application_name: str = "LangChain-CDBMongoVCore-VectorStore-Python",
        **kwargs: Any,
    ) -> AzureCosmosDBVectorSearch:
        """Creates an Instance of AzureCosmosDBVectorSearch
        from a Connection String

        Args:
            connection_string: The MongoDB vCore instance connection string
            namespace: The namespace (database.collection)
            embedding: The embedding utility
            application_name: The user agent for telemetry
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
        appname = application_name
        client: MongoClient = MongoClient(connection_string, appname=appname)
        db_name, collection_name = namespace.split(".")
        collection = client[db_name][collection_name]
        return cls(collection, embedding, **kwargs)

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

    def delete_index(self) -> None:
        """Deletes the index specified during instance construction if it exists"""
        if self.index_exists():
            self._collection.drop_index(self._index_name)
            # Raises OperationFailure on an error (e.g. trying to drop
            # an index that does not exist)

    def create_index(
        self,
        num_lists: int = 100,
        dimensions: int = 1536,
        similarity: CosmosDBSimilarityType = CosmosDBSimilarityType.COS,
        kind: str = "vector-ivf",
        m: int = 16,
        ef_construction: int = 64,
        max_degree: int = 32,
        l_build: int = 50,
    ) -> dict[str, Any]:
        """Creates an index using the index name specified at
            instance construction

        Setting the numLists parameter correctly is important for achieving
            good accuracy and performance.
            Since the vector store uses IVF as the indexing strategy,
            you should create the index only after you
            have loaded a large enough sample documents to ensure that the
            centroids for the respective buckets are
            faily distributed.

        We recommend that numLists is set to documentCount/1000 for up
            to 1 million documents
            and to sqrt(documentCount) for more than 1 million documents.
            As the number of items in your database grows, you should
            tune numLists to be larger
            in order to achieve good latency performance for vector search.

            If you're experimenting with a new scenario or creating a
            small demo, you can start with numLists
            set to 1 to perform a brute-force search across all vectors.
            This should provide you with the most
            accurate results from the vector search, however be aware that
            the search speed and latency will be slow.
            After your initial setup, you should go ahead and tune
            the numLists parameter using the above guidance.

        Args:
            kind: Type of vector index to create.
                Possible options are:
                    - vector-ivf
                    - vector-hnsw: available as a preview feature only,
                                   to enable visit https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/preview-features
                    - vector-diskann: available as a preview feature only
            num_lists: This integer is the number of clusters that the
                inverted file (IVF) index uses to group the vector data.
                We recommend that numLists is set to documentCount/1000
                for up to 1 million documents and to sqrt(documentCount)
                for more than 1 million documents.
                Using a numLists value of 1 is akin to performing
                brute-force search, which has limited performance
            dimensions: Number of dimensions for vector similarity.
                The maximum number of supported dimensions is 2000
            similarity: Similarity metric to use with the IVF index.

                Possible options are:
                    - CosmosDBSimilarityType.COS (cosine distance),
                    - CosmosDBSimilarityType.L2 (Euclidean distance), and
                    - CosmosDBSimilarityType.IP (inner product).
            m: The max number of connections per layer (16 by default, minimum
               value is 2, maximum value is 100). Higher m is suitable for datasets
               with high dimensionality and/or high accuracy requirements.
            ef_construction: the size of the dynamic candidate list for constructing
                            the graph (64 by default, minimum value is 4, maximum
                            value is 1000). Higher ef_construction will result in
                            better index quality and higher accuracy, but it will
                            also increase the time required to build the index.
                            ef_construction has to be at least 2 * m
            max_degree: Max number of neighbors.
                Default value is 32, range from 20 to 2048.
                Only vector-diskann search supports this for now.
            l_build: l value for index building.
                Default value is 50, range from 10 to 500.
                Only vector-diskann search supports this for now.
        Returns:
            An object describing the created index

        """
        # check the kind of vector search to be performed
        # prepare the command accordingly
        create_index_commands = {}
        if kind == CosmosDBVectorSearchType.VECTOR_IVF:
            create_index_commands = self._get_vector_index_ivf(
                kind, num_lists, similarity, dimensions
            )
        elif kind == CosmosDBVectorSearchType.VECTOR_HNSW:
            create_index_commands = self._get_vector_index_hnsw(
                kind, m, ef_construction, similarity, dimensions
            )
        elif kind == CosmosDBVectorSearchType.VECTOR_DISKANN:
            create_index_commands = self._get_vector_index_diskann(
                kind, max_degree, l_build, similarity, dimensions
            )

        # retrieve the database object
        current_database = self._collection.database

        # invoke the command from the database object
        create_index_responses: dict[str, Any] = current_database.command(
            create_index_commands
        )

        return create_index_responses

    def _get_vector_index_ivf(
        self, kind: str, num_lists: int, similarity: str, dimensions: int
    ) -> Dict[str, Any]:
        command = {
            "createIndexes": self._collection.name,
            "indexes": [
                {
                    "name": self._index_name,
                    "key": {self._embedding_key: "cosmosSearch"},
                    "cosmosSearchOptions": {
                        "kind": kind,
                        "numLists": num_lists,
                        "similarity": similarity,
                        "dimensions": dimensions,
                    },
                }
            ],
        }
        return command

    def _get_vector_index_hnsw(
        self, kind: str, m: int, ef_construction: int, similarity: str, dimensions: int
    ) -> Dict[str, Any]:
        command = {
            "createIndexes": self._collection.name,
            "indexes": [
                {
                    "name": self._index_name,
                    "key": {self._embedding_key: "cosmosSearch"},
                    "cosmosSearchOptions": {
                        "kind": kind,
                        "m": m,
                        "efConstruction": ef_construction,
                        "similarity": similarity,
                        "dimensions": dimensions,
                    },
                }
            ],
        }
        return command

    def _get_vector_index_diskann(
        self, kind: str, max_degree: int, l_build: int, similarity: str, dimensions: int
    ) -> Dict[str, Any]:
        command = {
            "createIndexes": self._collection.name,
            "indexes": [
                {
                    "name": self._index_name,
                    "key": {self._embedding_key: "cosmosSearch"},
                    "cosmosSearchOptions": {
                        "kind": kind,
                        "maxDegree": max_degree,
                        "lBuild": l_build,
                        "similarity": similarity,
                        "dimensions": dimensions,
                    },
                }
            ],
        }
        return command

    def create_filter_index(
        self,
        property_to_filter: str,
        index_name: str,
    ) -> dict[str, Any]:
        command = {
            "createIndexes": self._collection.name,
            "indexes": [
                {
                    "key": {property_to_filter: 1},
                    "name": index_name,
                }
            ],
        }
        # retrieve the database object
        current_database = self._collection.database

        # invoke the command from the database object
        create_index_responses: dict[str, Any] = current_database.command(command)
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
            {self._text_key: t, self._embedding_key: embedding, "metadata": m}
            for t, m, embedding in zip(texts, metadatas, embeddings)
        ]
        # insert the documents in Cosmos DB
        insert_result = self._collection.insert_many(to_insert)
        return insert_result.inserted_ids

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection: Optional[Collection] = None,
        **kwargs: Any,
    ) -> AzureCosmosDBVectorSearch:
        if collection is None:
            raise ValueError("Must provide 'collection' named parameter.")
        vectorstore = cls(collection, embedding, **kwargs)
        vectorstore.add_texts(texts, metadatas=metadatas)
        return vectorstore

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is None:
            raise ValueError("No document ids provided to delete.")

        for document_id in ids:
            self.delete_document_by_id(document_id)
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

    def _similarity_search_with_score(
        self,
        embeddings: List[float],
        k: int = 4,
        kind: CosmosDBVectorSearchType = CosmosDBVectorSearchType.VECTOR_IVF,
        pre_filter: Optional[Dict] = None,
        ef_search: int = 40,
        score_threshold: float = 0.0,
        l_search: int = 40,
        with_embedding: bool = False,
    ) -> List[Tuple[Document, float]]:
        """Returns a list of documents with their scores

        Args:
            embeddings: The query vector
            k: the number of documents to return
            kind: Type of vector index to create.
                Possible options are:
                    - vector-ivf
                    - vector-hnsw: available as a preview feature only,
                                   to enable visit https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/preview-features
                    - vector-diskann: available as a preview feature only
            ef_search: The size of the dynamic candidate list for search
                       (40 by default). A higher value provides better
                       recall at the cost of speed.
            score_threshold: (Optional[float], optional): Maximum vector distance
                between selected documents and the query vector. Defaults to None.
                Only vector-ivf search supports this for now.
            l_search: l value for index searching.
                Default value is 40, range from 10 to 10000.
                Only vector-diskann search supports this.

        Returns:
            A list of documents closest to the query vector
        """
        pipeline: List[dict[str, Any]] = []
        if kind == CosmosDBVectorSearchType.VECTOR_IVF:
            pipeline = self._get_pipeline_vector_ivf(embeddings, k, pre_filter)
        elif kind == CosmosDBVectorSearchType.VECTOR_HNSW:
            pipeline = self._get_pipeline_vector_hnsw(
                embeddings, k, ef_search, pre_filter
            )
        elif kind == CosmosDBVectorSearchType.VECTOR_DISKANN:
            pipeline = self._get_pipeline_vector_diskann(
                embeddings, k, l_search, pre_filter
            )

        cursor = self._collection.aggregate(pipeline)

        docs = []
        for res in cursor:
            score = res.pop("similarityScore")
            if score < score_threshold:
                continue
            document_object_field = res.pop("document")
            text = document_object_field.pop(self._text_key)
            metadata = document_object_field.pop("metadata", {})
            metadata["_id"] = document_object_field.pop(
                "_id"
            )  # '_id' is in new position
            if with_embedding:
                metadata[self._embedding_key] = document_object_field.pop(
                    self._embedding_key
                )

            docs.append((Document(page_content=text, metadata=metadata), score))
        return docs

    def _get_pipeline_vector_ivf(
        self, embeddings: List[float], k: int = 4, pre_filter: Optional[Dict] = None
    ) -> List[dict[str, Any]]:
        params = {
            "vector": embeddings,
            "path": self._embedding_key,
            "k": k,
        }
        if pre_filter:
            params["filter"] = pre_filter

        pipeline: List[dict[str, Any]] = [
            {
                "$search": {
                    "cosmosSearch": params,
                    "returnStoredSource": True,
                }
            },
            {
                "$project": {
                    "similarityScore": {"$meta": "searchScore"},
                    "document": "$$ROOT",
                }
            },
        ]
        return pipeline

    def _get_pipeline_vector_hnsw(
        self,
        embeddings: List[float],
        k: int = 4,
        ef_search: int = 40,
        pre_filter: Optional[Dict] = None,
    ) -> List[dict[str, Any]]:
        params = {
            "vector": embeddings,
            "path": self._embedding_key,
            "k": k,
            "efSearch": ef_search,
        }
        if pre_filter:
            params["filter"] = pre_filter

        pipeline: List[dict[str, Any]] = [
            {
                "$search": {
                    "cosmosSearch": params,
                }
            },
            {
                "$project": {
                    "similarityScore": {"$meta": "searchScore"},
                    "document": "$$ROOT",
                }
            },
        ]
        return pipeline

    def _get_pipeline_vector_diskann(
        self,
        embeddings: List[float],
        k: int = 4,
        l_search: int = 40,
        pre_filter: Optional[Dict] = None,
    ) -> List[dict[str, Any]]:
        params = {
            "vector": embeddings,
            "path": self._embedding_key,
            "k": k,
            "lSearch": l_search,
        }
        if pre_filter:
            params["filter"] = pre_filter

        pipeline: List[dict[str, Any]] = [
            {
                "$search": {
                    "cosmosSearch": params,
                }
            },
            {
                "$project": {
                    "similarityScore": {"$meta": "searchScore"},
                    "document": "$$ROOT",
                }
            },
        ]
        return pipeline

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        kind: CosmosDBVectorSearchType = CosmosDBVectorSearchType.VECTOR_IVF,
        pre_filter: Optional[Dict] = None,
        ef_search: int = 40,
        score_threshold: float = 0.0,
        l_search: int = 40,
        with_embedding: bool = False,
    ) -> List[Tuple[Document, float]]:
        embeddings = self._embedding.embed_query(query)
        docs = self._similarity_search_with_score(
            embeddings=embeddings,
            k=k,
            kind=kind,
            pre_filter=pre_filter,
            ef_search=ef_search,
            score_threshold=score_threshold,
            l_search=l_search,
            with_embedding=with_embedding,
        )
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        kind: CosmosDBVectorSearchType = CosmosDBVectorSearchType.VECTOR_IVF,
        pre_filter: Optional[Dict] = None,
        ef_search: int = 40,
        score_threshold: float = 0.0,
        l_search: int = 40,
        with_embedding: bool = False,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(
            query,
            k=k,
            kind=kind,
            pre_filter=pre_filter,
            ef_search=ef_search,
            score_threshold=score_threshold,
            l_search=l_search,
            with_embedding=with_embedding,
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        kind: CosmosDBVectorSearchType = CosmosDBVectorSearchType.VECTOR_IVF,
        pre_filter: Optional[Dict] = None,
        ef_search: int = 40,
        score_threshold: float = 0.0,
        l_search: int = 40,
        with_embedding: bool = False,
        **kwargs: Any,
    ) -> List[Document]:
        # Retrieves the docs with similarity scores
        # sorted by similarity scores in DESC order
        docs = self._similarity_search_with_score(
            embedding,
            k=fetch_k,
            kind=kind,
            pre_filter=pre_filter,
            ef_search=ef_search,
            score_threshold=score_threshold,
            l_search=l_search,
            with_embedding=with_embedding,
        )

        # Re-ranks the docs using MMR
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding),
            [doc.metadata[self._embedding_key] for doc, _ in docs],
            k=k,
            lambda_mult=lambda_mult,
        )
        mmr_docs = [docs[i][0] for i in mmr_doc_indexes]
        return mmr_docs

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        kind: CosmosDBVectorSearchType = CosmosDBVectorSearchType.VECTOR_IVF,
        pre_filter: Optional[Dict] = None,
        ef_search: int = 40,
        score_threshold: float = 0.0,
        l_search: int = 40,
        with_embedding: bool = False,
        **kwargs: Any,
    ) -> List[Document]:
        # compute the embeddings vector from the query string
        embeddings = self._embedding.embed_query(query)

        docs = self.max_marginal_relevance_search_by_vector(
            embeddings,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            kind=kind,
            pre_filter=pre_filter,
            ef_search=ef_search,
            score_threshold=score_threshold,
            l_search=l_search,
            with_embedding=with_embedding,
        )
        return docs

    def get_collection(self) -> Collection:
        return self._collection
