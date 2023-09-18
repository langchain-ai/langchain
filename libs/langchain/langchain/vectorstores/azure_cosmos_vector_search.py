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
    Union,
)

import numpy as np

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance
from enum import Enum


class CosmosDBAPI(Enum):
    NoSQL = 1
    MongoDB = 2
    Cassandra = 3
    Gremlin = 4
    Table = 5


if TYPE_CHECKING:
    from pymongo.collection import Collection

CosmosDBDocumentType = TypeVar("CosmosDBDocumentType", bound=Dict[str, Any])

logger = logging.getLogger(__name__)

DEFAULT_INSERT_BATCH_SIZE = 128


class AzureCosmosDBVectorSearch(VectorStore):
    """`MongoDB Atlas Vector Search` vector store.

       To use, you should have both:
       - the ``pymongo`` python package installed
       - a connection string associated with a MongoDB Atlas Cluster having deployed an
           Atlas Search index

       Example:
           .. code-block:: python

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
            api: CosmosDBAPI,
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
        self._api = api
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
        api = CosmosDBAPI.NoSQL
        return cls(collection, embedding, api, **kwargs)

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
        """Construct AzureCosmosDBVectorSearch wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Adds the documents to a provided Cosmos DB MongoDB API Vector Search index

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python
                from pymongo import MongoClient

                from langchain.vectorstores import AzureCosmosDBVectorSearch
                from langchain.embeddings import OpenAIEmbeddings

                mongo_client = MongoClient("<YOUR-CONNECTION-STRING>")
                collection = mongo_client["<db_name>"]["<collection_name>"]
                embeddings = OpenAIEmbeddings()
                vectorstore = AzureCosmosDBVectorSearch.from_texts(
                    texts,
                    embeddings,
                    metadatas=metadatas,
                    collection=collection
                )
        """
        if collection is None:
            raise ValueError("Must provide 'collection' named parameter.")
        vectorstore = cls(collection, embedding, **kwargs)
        vectorstore.add_texts(texts, metadatas=metadatas)
        return vectorstore

    def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:

        """Return Cosmos DB documents most similar to query.

                Use the vector search Operator available in Cosmos DB Search
        
                Args:
                    query: Text to look up documents similar to.
                    k: Optional Number of Documents to return. Defaults to 4.

                Returns:
                    List of Documents most similar to the query and score for each
                """"""Return MongoDB documents most similar to query.
        """
        # Compute the embeddings
        embeddings: list[float] = self._embedding.embed_query(query)

        pipeline = [
            {
                "$search": {
                    "cosmosSearch": {
                        "vector": embeddings,
                        "path": self._embedding_key,
                        "k": k,
                    },
                    "returnStoredSource": False,
                }
            }
        ]

        cursor = self._collection.aggregate(pipeline)

        docs = []

        for res in cursor:
            text = res.pop(self._text_key)
            single_doc = Document(page_content=text, metadata=res)
            docs.append(single_doc)
        return docs
