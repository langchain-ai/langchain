from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union, Mapping,
)

import numpy as np

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.runnable.configurable import StrEnum
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance


class CosmosDBSimilarityType(StrEnum):  # Before Python 3.11 native StrEnum is not available
    COS = 'COS'  # CosineSimilarity
    L2 = 'L2'  # Euclidean distance
    IP = 'IP'  # inner - product


if TYPE_CHECKING:
    from pymongo.collection import Collection

CosmosDBDocumentType = TypeVar("CosmosDBDocumentType", bound=Dict[str, Any])

logger = logging.getLogger(__name__)

DEFAULT_INSERT_BATCH_SIZE = 128


class AzureCosmosDBVectorSearch(VectorStore):
    """`Cosmos DB for MongoDB vCore` vector store.

       To use, you should have both:
       - the ``pymongo`` python package installed
       - a connection string associated with a MongoDB VCore Cluster having deployed an
           Atlas Search index

       Example:
           . code-block:: python

               from langchain.vectorstores import AzureCosmosDBVectorSearch
               from langchain.embeddings.openai import OpenAIEmbeddings
               from pymongo import MongoClient

               mongo_client = MongoClient("<YOUR-CONNECTION-STRING>")
               collection = mongo_client["<db_name>"]["<collection_name>"]
               embeddings = OpenAIEmbeddings()
               vectorstore = AzureCosmosDBVectorSearch(collection, embeddings)
       """

    def __init__(
            self,
            collection: Collection[CosmosDBDocumentType],
            embedding: Embeddings,
            *,
            index_name: str = "vectorSearchIndex",
            text_key: str = "textContent",
            embedding_key: str = "vectorContent",
    ):
        """
        Args:
            collection: MongoDB collection to add the texts to.
            embedding: Text embedding model to use.
            text_key: MongoDB field that will contain the text for each
                document.
            embedding_key: MongoDB field that will contain the embedding for
                each document.
            index_name: Name of the Atlas Search index.
        """
        self._collection = collection
        self._embedding = embedding
        self._index_name = index_name
        self._text_key = text_key
        self._embedding_key = embedding_key

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @classmethod
    def from_connection_string(
            cls,
            connection_string: str,
            namespace: str,
            embedding: Embeddings,
            **kwargs: Any,
    ) -> AzureCosmosDBVectorSearch:
        """

        Args:
            connection_string: The MongoDB vCore connection string
            namespace: The namespace
            embedding: The Embedding utilty
            **kwargs: Dynamic arguments

        Returns:

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

    def index_exists(self, index_name):

        cursor = self._collection.list_indexes()

        for res in cursor:
            current_index_name = res.pop("name")
            if current_index_name == index_name:
                return True

        return False

    def delete_index(self) -> object:
        if self.index_exists(self._index_name):
            self._collection.drop_index(self._index_name)
            # Raises OperationFailure on an error (e.g. trying to drop
            # an index that does not exist)

    def create_index(self, num_lists: int = 100, dimensions: int = 1536,
                     similarity: CosmosDBSimilarityType = CosmosDBSimilarityType.COS):

        # prepare the command
        create_index_commands = {
            "createIndexes": self._collection.name,
            "indexes": [
                {
                    "name": self._index_name,
                    "key": {
                        "vectorContent": "cosmosSearch"
                    },
                    "cosmosSearchOptions": {
                        "kind": 'vector-ivf',
                        "numLists": num_lists,
                        "similarity": similarity,
                        "dimensions": dimensions
                    }
                }
            ]
        }

        # retrieve the database object
        current_database = self._collection.database

        # invoke the command from the database object
        create_index_responses: dict[str, Any] = current_database.command(create_index_commands)

        return create_index_responses

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[Dict[str, Any]]] = None,
            **kwargs: Any,
    ) -> List:
        """Let's Run more text content through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
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
        if not texts:
            return []
        # Embed and create the documents
        embeddings = self._embedding.embed_documents(texts)
        to_insert = [
            {self._text_key: t, self._embedding_key: embedding, **m}
            for t, m, embedding in zip(texts, metadatas, embeddings)
        ]
        # insert the documents in MongoDB Atlas
        insert_result = self._collection.insert_many(to_insert)  # type: ignore
        return insert_result.inserted_ids

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            collection: Optional[Collection[CosmosDBDocumentType]] = None,
            **kwargs: Any,
    ) -> AzureCosmosDBVectorSearch:
        if collection is None:
            raise ValueError("Must provide 'collection' named parameter.")
        vectorstore = cls(collection, embedding, **kwargs)
        vectorstore.add_texts(texts, metadatas=metadatas)
        return vectorstore

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is None:
            raise ValueError("No ids provided to delete.")

        for document_id in ids:
            self.delete_by_document_id(document_id)
        return True

    def delete_document_by_id(self, document_id: str):
        self._collection.delete_one({"_id": document_id})

    def _similarity_search_with_score(
            self, embeddings: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:

        pipeline = [
            {
                "$search": {
                    "cosmosSearch": {
                        "vector": embeddings,
                        "path": self._embedding_key,
                        "k": k,
                    },
                    "returnStoredSource": True,
                }
            },
            {
                "$project": {
                    "similarityScore": {
                        "$meta": "searchScore"
                    },
                    "document": "$$ROOT"
                }
            }
        ]

        cursor = self._collection.aggregate(pipeline)

        docs = []

        for res in cursor:
            score = res.pop("similarityScore")
            document_object_field: Mapping[str, Any] = res.pop("document")
            text = document_object_field.pop(self._text_key)
            docs.append((Document(page_content=text, metadata=res), score))

        return docs

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        embeddings = self._embedding.embed_query(query)
        docs = self._similarity_search_with_score(embeddings=embeddings, k=k)
        return docs

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            **kwargs: Any,
    ) -> List[Document]:

        # Retrieves the docs with similarity scores sorted by similarity scores in DESC order
        docs = self._similarity_search_with_score(embedding, k=fetch_k)

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
            **kwargs: Any,
    ) -> List[Document]:

        embeddings = self._embedding.embed_query(query)

        docs = self.max_marginal_relevance_search_by_vector(query, embeddings, k=k, fetch_k=fetch_k,
                                                            lambda_mult=lambda_mult)
        return docs
