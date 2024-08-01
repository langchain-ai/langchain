from __future__ import annotations

import logging
from importlib.metadata import version
from typing import (
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
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.driver_info import DriverInfo
from pymongo.errors import CollectionInvalid

from langchain_mongodb.index import (
    create_vector_search_index,
    update_vector_search_index,
)
from langchain_mongodb.utils import (
    make_serializable,
    maximal_marginal_relevance,
    oid_to_str,
    str_to_oid,
)

MongoDBDocumentType = TypeVar("MongoDBDocumentType", bound=Dict[str, Any])
VST = TypeVar("VST", bound=VectorStore)

logger = logging.getLogger(__name__)

DEFAULT_INSERT_BATCH_SIZE = 100_000


class MongoDBAtlasVectorSearch(VectorStore):
    """`MongoDB Atlas Vector Search` vector store.

    To use, you should have both:
    - the ``pymongo`` python package installed
    - a connection string associated with a MongoDB Atlas Cluster having deployed an
        Atlas Search index

    Example:
        .. code-block:: python

            from langchain_mongodb import MongoDBAtlasVectorSearch
            from langchain_openai import OpenAIEmbeddings
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
                defaults to 'text'
            embedding_key: MongoDB field that will contain the embedding for
                each document.
                defaults to 'embedding'
            index_name: Name of the Atlas Search index.
                defaults to 'default'
            relevance_score_fn: The similarity score used for the index.
                defaults to 'cosine'
            Currently supported: 'euclidean', 'cosine', and 'dotProduct'.
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
        scoring: dict[str, Callable] = {
            "euclidean": self._euclidean_relevance_score_fn,
            "dotProduct": self._max_inner_product_relevance_score_fn,
            "cosine": self._cosine_relevance_score_fn,
        }
        if self._relevance_score_fn in scoring:
            return scoring[self._relevance_score_fn]
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
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts, create embeddings, and add to the Collection and index.

        Important notes on ids:
            - If _id or id is a key in the metadatas dicts, one must
                pop them and provide as separate list.
            - They must be unique.
            - If they are not provided, the VectorStore will create unique ones,
                stored as bson.ObjectIds internally, and strings in Langchain.
                These will appear in Document.metadata with key, '_id'.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique ids that will be used as index in VectorStore.
                See note on ids.

        Returns:
            List of ids added to the vectorstore.
        """

        # Check to see if metadata includes ids
        if metadatas is not None and (
            metadatas[0].get("_id") or metadatas[0].get("id")
        ):
            logger.warning(
                "_id or id key found in metadata. "
                "Please pop from each dict and input as separate list."
                "Retrieving methods will include the same id as '_id' in metadata."
            )

        texts_batch = texts
        _metadatas: Union[List, Generator] = metadatas or ({} for _ in texts)
        metadatas_batch = _metadatas

        result_ids = []
        batch_size = kwargs.get("batch_size", DEFAULT_INSERT_BATCH_SIZE)
        if batch_size:
            texts_batch = []
            metadatas_batch = []
            size = 0
            i = 0
            for j, (text, metadata) in enumerate(zip(texts, _metadatas)):
                size += len(text) + len(metadata)
                texts_batch.append(text)
                metadatas_batch.append(metadata)
                if (j + 1) % batch_size == 0 or size >= 47_000_000:
                    if ids:
                        batch_res = self.bulk_embed_and_insert_texts(
                            texts_batch, metadatas_batch, ids[i : j + 1]
                        )
                    else:
                        batch_res = self.bulk_embed_and_insert_texts(
                            texts_batch, metadatas_batch
                        )
                    result_ids.extend(batch_res)
                    texts_batch = []
                    metadatas_batch = []
                    size = 0
                    i = j + 1
        if texts_batch:
            if ids:
                batch_res = self.bulk_embed_and_insert_texts(
                    texts_batch, metadatas_batch, ids[i : j + 1]
                )
            else:
                batch_res = self.bulk_embed_and_insert_texts(
                    texts_batch, metadatas_batch
                )
            result_ids.extend(batch_res)
        return result_ids

    def bulk_embed_and_insert_texts(
        self,
        texts: Union[List[str], Iterable[str]],
        metadatas: Union[List[dict], Generator[dict, Any, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Bulk insert single batch of texts, embeddings, and optionally ids.

        See add_texts for additional details.
        """
        if not texts:
            return []
        # Compute embedding vectors
        embeddings = self._embedding.embed_documents(texts)  # type: ignore
        if ids:
            to_insert = [
                {
                    "_id": str_to_oid(i),
                    self._text_key: t,
                    self._embedding_key: embedding,
                    **m,
                }
                for i, t, m, embedding in zip(ids, texts, metadatas, embeddings)
            ]
        else:
            to_insert = [
                {self._text_key: t, self._embedding_key: embedding, **m}
                for t, m, embedding in zip(texts, metadatas, embeddings)
            ]
        # insert the documents in MongoDB Atlas
        insert_result = self._collection.insert_many(to_insert)  # type: ignore
        return [oid_to_str(_id) for _id in insert_result.inserted_ids]

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        batch_size: int = DEFAULT_INSERT_BATCH_SIZE,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            ids: Optional list of unique ids that will be used as index in VectorStore.
                See note on ids in add_texts.
            batch_size: Number of documents to insert at a time.
                Tuning this may help with performance and sidestep MongoDB limits.

        Returns:
            List of IDs of the added texts.
        """
        n_docs = len(documents)
        if ids:
            assert len(ids) == n_docs, "Number of ids must equal number of documents."
        result_ids = []
        start = 0
        for end in range(batch_size, n_docs + batch_size, batch_size):
            texts, metadatas = zip(
                *[(doc.page_content, doc.metadata) for doc in documents[start:end]]
            )
            if ids:
                result_ids.extend(
                    self.bulk_embed_and_insert_texts(
                        texts=texts, metadatas=metadatas, ids=ids[start:end]
                    )
                )
            else:
                result_ids.extend(
                    self.bulk_embed_and_insert_texts(texts=texts, metadatas=metadatas)
                )
            start = end
        return result_ids

    def _similarity_search_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        include_embedding: bool = False,
        include_ids: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Core implementation."""
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

        # Exclude the embedding key from the return payload
        if not include_embedding:
            pipeline.append({"$project": {self._embedding_key: 0}})

        if post_filter_pipeline is not None:
            pipeline.extend(post_filter_pipeline)
        cursor = self._collection.aggregate(pipeline)  # type: ignore[arg-type]
        docs = []

        for res in cursor:
            text = res.pop(self._text_key)
            score = res.pop("score")
            make_serializable(res)
            docs.append((Document(page_content=text, metadata=res), score))
        return docs

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        **kwargs: Any,
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
            **kwargs,
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
            **kwargs,
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
        return self.max_marginal_relevance_search_by_vector(
            embedding=query_embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict]] = None,
        collection: Optional[Collection[MongoDBDocumentType]] = None,
        ids: Optional[List[str]] = None,
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

                from langchain_mongodb import MongoDBAtlasVectorSearch
                from langchain_openai import OpenAIEmbeddings

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
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)
        return vectorstore

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents from VectorStore by ids.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments passed to Collection.delete_many()

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        filter = {}
        if ids:
            oids = [str_to_oid(i) for i in ids]
            filter = {"_id": {"$in": oids}}
        return self._collection.delete_many(filter=filter, **kwargs).acknowledged

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        return await run_in_executor(None, self.delete, ids=ids, **kwargs)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> List[Document]:  # type: ignore
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            pre_filter: (Optional) dictionary of argument(s) to prefilter on document
                fields.
            post_filter_pipeline: (Optional) pipeline of MongoDB aggregation stages
                following the vectorSearch stage.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        docs = self._similarity_search_with_score(
            embedding,
            k=fetch_k,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            include_embedding=kwargs.pop("include_embedding", True),
            **kwargs,
        )
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding),
            [doc.metadata[self._embedding_key] for doc, _ in docs],
            k=k,
            lambda_mult=lambda_mult,
        )
        mmr_docs = [docs[i][0] for i in mmr_doc_indexes]
        return mmr_docs

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search_by_vector,
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    def create_vector_search_index(
        self,
        dimensions: int,
        filters: Optional[List[Dict[str, str]]] = None,
        update: bool = False,
    ) -> None:
        """Creates a MongoDB Atlas vectorSearch index for the VectorStore

        Note**: This method may fail as it requires a MongoDB Atlas with
        these pre-requisites:
            - M10 cluster or higher
            - https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/#prerequisites

        Args:
            dimensions (int): Number of dimensions in embedding
            filters (Optional[List[Dict[str, str]]], optional): additional filters
            for index definition.
                Defaults to None.
            update (bool, optional): Updates existing vectorSearch index.
                Defaults to False.
        """
        try:
            self._collection.database.create_collection(self._collection.name)
        except CollectionInvalid:
            pass

        index_operation = (
            update_vector_search_index if update else create_vector_search_index
        )

        index_operation(
            collection=self._collection,
            index_name=self._index_name,
            dimensions=dimensions,
            path=self._embedding_key,
            similarity=self._relevance_score_fn,
            filters=filters or [],
        )  # type: ignore [operator]
