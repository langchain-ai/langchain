from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, List, Optional, Tuple, Dict

from pymongo import MongoClient

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

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
        embedding_function: Callable,
        client: MongoClient,
        namespace: str,
        index_name: str = "default"
    ):
        self._client = client
        db_name, collection_name = namespace.split('.')
        self._collection = client[db_name][collection_name]
        self._index_name = index_name
        self._embedding_function = embedding_function

    @classmethod
    def from_connection_string(
        cls,
        embedding: Embeddings,
        connection_string: str,
        namespace: str,
        index_name: str = "default"
    ) -> MongoDBAtlasVectorSearch:
        client = MongoClient(connection_string)
        return cls(embedding.embed_query, client, namespace, index_name)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        namespace: Optional[str] = None,
        text_key: str = "text",
        embedding_key: str = "embedding",
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            namespace: Optional MongoDB namespace to add the texts to.
            text_key: Optional MongoDB field that will contain the text for each document.
            embedding_key: Optional MongoDB field that will contain the embedding for each document.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        collection = self._collection
        if namespace is not None:
            db_name, collection_name = namespace.split('.')
            collection = self._client[db_name][collection_name]

        # Embed and create the documents
        docs = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            embedding = self._embedding_function(text)
            docs.append({**metadata, text_key: text, embedding_key: embedding})

        # insert in MongoDB Atlas
        insert_result = collection.insert_many(docs)
        return insert_result.inserted_ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        namespace: Optional[str] = None,
        index_name: Optional[str] = None,
        text_key: str = "text",
        embedding_key: str = "embedding"
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
            namespace: Optional Namespace to search in. Default will search in the namespace passed in the constructor.
            index_name: Optional Atlas Search Index to use. Default will be the index passed in the constructor.
            text_key: Optional MongoDB field that contains the text for each document.
            embedding_key: Optional MongoDB field that contains the embedding for each document.

        Returns:
            List of Documents most similar to the query and score for each
        """
        collection = self._collection
        if namespace is not None:
            db_name, collection_name = namespace.split('.')
            collection = self._client[db_name][collection_name]
        index_name = index_name or self._index_name
        query_vector = self._embedding_function(query)
        pipeline = [
            {
                "$search": {
                    "index": index_name,
                    "knnBeta": {
                        "vector": query_vector,
                        "path": embedding_key,
                        "k": k,
                        "filter": pre_filter or {}
                    }
                }
            },
            {
                "$project": {
                    "score": {"$meta": "searchScore"},
                    embedding_key: 0
                }
            }
        ]
        if post_filter_pipeline is not None:
            pipeline.extend(post_filter_pipeline)
        cursor = collection.aggregate(pipeline)
        docs = []
        for document in cursor:
            text = document.pop(text_key)
            score = document.pop("score")
            metadata = document
            docs.append((Document(page_content=text, metadata=metadata), score))
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        namespace: Optional[str] = None,
        index_name: Optional[str] = None,
        text_key: str = "text",
        embedding_key: str = "embedding"
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
            namespace: Optional Namespace to search in. Default will search in the namespace passed in the constructor.
            index_name: Optional Atlas Search Index to use. Default will be the index passed in the constructor.
            text_key: Optional MongoDB field that contains the text for each document.
            embedding_key: Optional MongoDB field that contains the embedding for each document.

        Returns:
            List of Documents most similar to the query and score for each
        """
        docs_and_scores = self.similarity_search_with_score(
            query,
            k=k,
            pre_filter=pre_filter,
            post_filter_pipeline=namespace,
            namespace=namespace,
            index_name=pre_filter,
            text_key=text_key,
            embedding_key=embedding_key
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        text_key: str = "text",
        embedding_key: str = "embedding",
        **kwargs: Any
    ) -> MongoDBAtlasVectorSearch:
        """Construct MongoDBAtlasVectorSearch wrapper from raw documents.

        This is a user friendly interface that:
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
                    metadatas,
                    client=client,
                    namespace=namespace
                )
        """
        try:
            client, namespace = kwargs["client"], kwargs["namespace"]
        except KeyError:
            raise ValueError("Please provide a MongoDB client and a namespace to query the documents")
        index_name = kwargs.get("index_name", "default")
        db_name, collection_name = namespace.split('.')
        collection = client[db_name][collection_name]
        # create embeddings
        embeds = embedding.embed_documents(texts)
        # create the documents
        docs = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            docs.append({**metadata, text_key: text, embedding_key: embeds[i]})
        # insert in MongoDB Atlas
        collection.insert_many(docs)
        return cls(embedding.embed_query, client, namespace, index_name)
