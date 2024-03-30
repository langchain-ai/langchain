from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
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
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from pymongo.collection import Collection

MongoDBDocumentType = TypeVar("MongoDBDocumentType", bound=Dict[str, Any])

logger = logging.getLogger(__name__)

DEFAULT_INSERT_BATCH_SIZE = 100


@deprecated(
    since="0.0.25",
    removal="0.2.0",
    alternative_import="langchain_mongodb.MongoDBAtlasVectorSearch",
)
class MongoDBAtlasVectorSearch(VectorStore):
    """`MongoDB Atlas Vector Search` vector store.

    To use, you should have both:
    - the ``pymongo`` python package installed
    - a connection string associated with a MongoDB Atlas Cluster having deployed an
        Atlas Search index

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import MongoDBAtlasVectorSearch
            from langchain_community.embeddings.openai import OpenAIEmbeddings
            from pymongo import MongoClient

            mongo_client = MongoClient("<YOUR-CONNECTION-STRING>")
            collection = mongo_client["<db_name>"]["<collection_name>"]
            embeddings = OpenAIEmbeddings()
            vectorstore = MongoDBAtlasVectorSearch(collection, embeddings)
    """

    def __init__(
        self,
        collection: Collection[MongoDBDocumentType],
        embedding: Embeddings,
        *,
        index_name: str = "default",
        text_key: str = "text",
        embedding_key: str = "embedding",
        relevance_score_fn: str = "cosine",
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
            relevance_score_fn: The similarity score used for the index.
            Currently supported: Euclidean, cosine, and dot product.
        """
        self._collection = collection
        self._embedding = embedding
        self._index_name = index_name
        self._text_key = text_key
        self._embedding_key = embedding_key
        self._relevance_score_fn = relevance_score_fn

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self._relevance_score_fn == "euclidean":
            return self._euclidean_relevance_score_fn
        elif self._relevance_score_fn == "dotProduct":
            return self._max_inner_product_relevance_score_fn
        elif self._relevance_score_fn == "cosine":
            return self._cosine_relevance_score_fn
        else:
            raise NotImplementedError(
                f"No relevance score function for ${self._relevance_score_fn}"
            )

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        namespace: str,
        embedding: Embeddings,
        **kwargs: Any,
    ) -> MongoDBAtlasVectorSearch:
        """Construct a `MongoDB Atlas Vector Search` vector store
        from a MongoDB connection URI.

        Args:
            connection_string: A valid MongoDB connection URI.
            namespace: A valid MongoDB namespace (database and collection).
            embedding: The text embedding model to use for the vector store.

        Returns:
            A new MongoDBAtlasVectorSearch instance.

        """
        try:
            from importlib.metadata import version

            from pymongo import MongoClient
            from pymongo.driver_info import DriverInfo
        except ImportError:
            raise ImportError(
                "Could not import pymongo, please install it with "
                "`pip install pymongo`."
            )
        client: MongoClient = MongoClient(
            connection_string,
            driver=DriverInfo(name="Langchain", version=version("langchain")),
        )
        db_name, collection_name = namespace.split(".")
        collection = client[db_name][collection_name]
        return cls(collection, embedding, **kwargs)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List:
        """Run more texts through the embeddings and add to the vectorstore.

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

    def _similarity_search_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
    ) -> List[Tuple[Document, float]]:
        params = {
            "queryVector": embedding,
            "path": self._embedding_key,
            "numCandidates": k * 10,
            "limit": k,
            "index": self._index_name,
        }
        if pre_filter:
            params["filter"] = pre_filter
        query = {"$vectorSearch": params}

        pipeline = [
            query,
            {"$set": {"score": {"$meta": "vectorSearchScore"}}},
        ]
        if post_filter_pipeline is not None:
            pipeline.extend(post_filter_pipeline)
        cursor = self._collection.aggregate(pipeline)  # type: ignore[arg-type]
        docs = []
        for res in cursor:
            text = res.pop(self._text_key)
            score = res.pop("score")
            docs.append((Document(page_content=text, metadata=res), score))
        return docs

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return MongoDB documents most similar to the given query and their scores.

        Uses the vectorSearch operator available in MongoDB Atlas Search.
        For more: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/

        Args:
            query: Text to look up documents similar to.
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: (Optional) dictionary of argument(s) to prefilter document
                fields on.
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages
                following the vectorSearch stage.

        Returns:
            List of documents most similar to the query and their scores.
        """
        embedding = self._embedding.embed_query(query)
        docs = self._similarity_search_with_score(
            embedding,
            k=k,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
        )
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return MongoDB documents most similar to the given query.

        Uses the vectorSearch operator available in MongoDB Atlas Search.
        For more: https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/

        Args:
            query: Text to look up documents similar to.
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: (Optional) dictionary of argument(s) to prefilter document
                fields on.
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages
                following the vectorSearch stage.

        Returns:
            List of documents most similar to the query and their scores.
        """
        additional = kwargs.get("additional")
        docs_and_scores = self.similarity_search_with_score(
            query,
            k=k,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
        )

        if additional and "similarity_score" in additional:
            for doc, score in docs_and_scores:
                doc.metadata["score"] = score
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: (Optional) number of documents to return. Defaults to 4.
            fetch_k: (Optional) number of documents to fetch before passing to MMR
                algorithm. Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            pre_filter: (Optional) dictionary of argument(s) to prefilter on document
                fields.
            post_filter_pipeline: (Optional) pipeline of MongoDB aggregation stages
                following the vectorSearch stage.
        Returns:
            List of documents selected by maximal marginal relevance.
        """
        query_embedding = self._embedding.embed_query(query)
        docs = self._similarity_search_with_score(
            query_embedding,
            k=fetch_k,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
        )
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(query_embedding),
            [doc.metadata[self._embedding_key] for doc, _ in docs],
            k=k,
            lambda_mult=lambda_mult,
        )
        mmr_docs = [docs[i][0] for i in mmr_doc_indexes]
        return mmr_docs

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict]] = None,
        collection: Optional[Collection[MongoDBDocumentType]] = None,
        **kwargs: Any,
    ) -> MongoDBAtlasVectorSearch:
        """Construct a `MongoDB Atlas Vector Search` vector store from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Adds the documents to a provided MongoDB Atlas Vector Search index
                (Lucene)

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python
                from pymongo import MongoClient

                from langchain_community.vectorstores import MongoDBAtlasVectorSearch
                from langchain_community.embeddings import OpenAIEmbeddings

                mongo_client = MongoClient("<YOUR-CONNECTION-STRING>")
                collection = mongo_client["<db_name>"]["<collection_name>"]
                embeddings = OpenAIEmbeddings()
                vectorstore = MongoDBAtlasVectorSearch.from_texts(
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
