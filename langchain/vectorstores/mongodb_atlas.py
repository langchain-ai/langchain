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

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

if TYPE_CHECKING:
    from pymongo.collection import Collection

MongoDBDocumentType = TypeVar("MongoDBDocumentType", bound=Dict[str, Any])

logger = logging.getLogger(__name__)

DEFAULT_INSERT_BATCH_SIZE = 100


class MongoDBAtlasVectorSearch(VectorStore):
    """Wrapper around MongoDB Atlas Vector Search.

    To use, you should have both:
    - the ``pymongo`` python package installed
    - a connection string associated with a MongoDB Atlas Cluster having deployed an
        Atlas Search index

    Example:
        .. code-block:: python

            from langchain.vectorstores import MongoDBAtlasVectorSearch
            from langchain.embeddings.openai import OpenAIEmbeddings
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
    ):
        """
        Args:
            collection: MongoDB collection to add the texts to.
            embedding: Text embedding model to use.
            text_key: MongoDB field that will contain the text for each
                document.
            embedding_key: MongoDB field that will contain the embedding for
                each document.
        """
        self._collection = collection
        self._embedding = embedding
        self._index_name = index_name
        self._text_key = text_key
        self._embedding_key = embedding_key

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        namespace: str,
        embedding: Embeddings,
        **kwargs: Any,
    ) -> MongoDBAtlasVectorSearch:
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
        insert_result = self._collection.insert_many(to_insert)
        return insert_result.inserted_ids

    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: int = 4,
        pre_filter: Optional[dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return MongoDB documents most similar to query, along with scores.

        Use the knnBeta Operator available in MongoDB Atlas Search
        This feature is in early access and available only for evaluation purposes, to
        validate functionality, and to gather feedback from a small closed group of
        early access users. It is not recommended for production deployments as we
        may introduce breaking changes.
        For more: https://www.mongodb.com/docs/atlas/atlas-search/knn-beta

        Args:
            query: Text to look up documents similar to.
            k: Optional Number of Documents to return. Defaults to 4.
            pre_filter: Optional Dictionary of argument(s) to prefilter on document
                fields.
            post_filter_pipeline: Optional Pipeline of MongoDB aggregation stages
                following the knnBeta search.

        Returns:
            List of Documents most similar to the query and score for each
        """
        knn_beta = {
            "vector": self._embedding.embed_query(query),
            "path": self._embedding_key,
            "k": k,
        }
        if pre_filter:
            knn_beta["filter"] = pre_filter
        pipeline = [
            {
                "$search": {
                    "index": self._index_name,
                    "knnBeta": knn_beta,
                }
            },
            {"$project": {"score": {"$meta": "searchScore"}, self._embedding_key: 0}},
        ]
        if post_filter_pipeline is not None:
            pipeline.extend(post_filter_pipeline)
        cursor = self._collection.aggregate(pipeline)
        docs = []
        for res in cursor:
            text = res.pop(self._text_key)
            score = res.pop("score")
            docs.append((Document(page_content=text, metadata=res), score))
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return MongoDB documents most similar to query.

        Use the knnBeta Operator available in MongoDB Atlas Search
        This feature is in early access and available only for evaluation purposes, to
        validate functionality, and to gather feedback from a small closed group of
        early access users. It is not recommended for production deployments as we may
        introduce breaking changes.
        For more: https://www.mongodb.com/docs/atlas/atlas-search/knn-beta

        Args:
            query: Text to look up documents similar to.
            k: Optional Number of Documents to return. Defaults to 4.
            pre_filter: Optional Dictionary of argument(s) to prefilter on document
                fields.
            post_filter_pipeline: Optional Pipeline of MongoDB aggregation stages
                following the knnBeta search.

        Returns:
            List of Documents most similar to the query and score for each
        """
        docs_and_scores = self.similarity_search_with_score(
            query,
            k=k,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection: Optional[Collection[MongoDBDocumentType]] = None,
        **kwargs: Any,
    ) -> MongoDBAtlasVectorSearch:
        """Construct MongoDBAtlasVectorSearch wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Adds the documents to a provided MongoDB Atlas Vector Search index
                (Lucene)

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python
                from pymongo import MongoClient

                from langchain.vectorstores import MongoDBAtlasVectorSearch
                from langchain.embeddings import OpenAIEmbeddings

                client = MongoClient("<YOUR-CONNECTION-STRING>")
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
        vecstore = cls(collection, embedding, **kwargs)
        vecstore.add_texts(texts, metadatas=metadatas)
        return vecstore
