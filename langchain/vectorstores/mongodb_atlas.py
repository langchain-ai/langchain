from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

if TYPE_CHECKING:
    from pymongo import MongoClient

logger = logging.getLogger(__name__)


class MongoDBAtlasVectorSearch(VectorStore):
    """Wrapper around MongoDB Atlas Vector Search.

    To use, you should have both:
    - the ``pymongo`` python package installed
    - a connection string associated with a MongoDB Atlas Cluster having deployed an Atlas Search index

    Example:
        .. code-block:: python

            from langchain.vectorstores import MongoDBAtlasVectorSearch
            from langchain.embeddings.openai import OpenAIEmbeddings
            from pymongo import MongoClient

            mongo_client = MongoClient("<YOUR-CONNECTION-STRING>")
            namespace = "<db_name>.<collection_name>"
            embeddings = OpenAIEmbeddings()
            vectorstore = MongoDBAtlasVectorSearch(embeddings.embed_query, mongo_client, namespace)
    """

    def __init__(
        self,
        client: MongoClient,
        namespace: str,
        embedding: Embeddings,
        *,
        index_name: str = "default",
        text_key: str = "text",
        embedding_key: str = "embedding,",
    ):
        """
        Args:
            client: MongoDB client.
            namespace: MongoDB namespace to add the texts to.
            embedding: Text embedding model to use.
            text_key: MongoDB field that will contain the text for each
                document.
            embedding_key: MongoDB field that will contain the embedding for
                each document.
        """
        self._client = client
        db_name, collection_name = namespace.split(".")
        self._collection = client[db_name][collection_name]
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
        client = MongoClient(connection_string)
        return cls(client, namespace, embedding, **kwargs)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        # Embed and create the documents
        docs = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            embedding = self._embedding.embed_documents([text])[0]
            doc = {self._text_key: text, self._embedding_key: embedding, **metadata}
            docs.append(doc)

        # insert in MongoDB Atlas
        insert_result = self._collection.insert_many(docs)
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
        This feature is in early access and available only for evaluation purposes, to validate functionality, and
        to gather feedback from a small closed group of early access users. It is not recommended for production
        deployments as we may introduce breaking changes.
        https://www.mongodb.com/docs/atlas/atlas-search/knn-beta

        Args:
            query: Text to look up documents similar to.
            k: Optional Number of Documents to return. Defaults to 4.
            pre_filter: Optional Dictionary of argument(s) to prefilter on document fields.
            post_filter_pipeline: Optional Pipeline of MongoDB aggregation stages following the knnBeta search.

        Returns:
            List of Documents most similar to the query and score for each
        """
        pipeline = [
            {
                "$search": {
                    "index": self._index_name,
                    "knnBeta": {
                        "vector": self._embedding.embed_query(query),
                        "path": self._embedding_key,
                        "k": k,
                        "filter": pre_filter or {},
                    },
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
        This feature is in early access and available only for evaluation purposes, to validate functionality, and
        to gather feedback from a small closed group of early access users. It is not recommended for production
        deployments as we may introduce breaking changes.
        https://www.mongodb.com/docs/atlas/atlas-search/knn-beta

        Args:
            query: Text to look up documents similar to.
            k: Optional Number of Documents to return. Defaults to 4.
            pre_filter: Optional Dictionary of argument(s) to prefilter on document fields.
            post_filter_pipeline: Optional Pipeline of MongoDB aggregation stages following the knnBeta search.

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
        connection_string: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> MongoDBAtlasVectorSearch:
        """Construct MongoDBAtlasVectorSearch wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Adds the documents to a provided MongoDB Atlas Vector Search index (Lucene)

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain.vectorstores import MongoDBAtlasVectorSearch
                from langchain.embeddings import OpenAIEmbeddings
                from pymongo import MongoClient

                mongo_client = MongoClient("<YOUR-CONNECTION-STRING>")
                namespace = "<db_name>.<collection_name>"
                embeddings = OpenAIEmbeddings()
                vectorstore = MongoDBAtlasVectorSearch.from_texts(
                    texts,
                    embeddings,
                    meadatas=metadatas,
                    client=client,
                    namespace=namespace
                )
        """
        if not connection_string or not namespace:
            raise ValueError(
                "Must provide 'connection_string' and 'namespace' named parameters."
            )
        vecstore = cls.from_connection_string(
            connection_string, namespace, embedding, **kwargs
        )
        vecstore.add_texts(texts, metadatas=metadatas)
        return vecstore
