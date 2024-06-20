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

from langchain_mongodb.pipelines import (
    combine_pipelines,
    final_hybrid_stage,
    reciprocal_rank_stage,
    text_search_stage,
    vector_search_stage,
)
from langchain_mongodb.utils import make_serializable, maximal_marginal_relevance

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
        vector_index_name: str = "vector_index",
        text_key: str = "text",
        embedding_key: str = "embedding",
        relevance_score_fn: str = "cosine",
        fulltext_index_name: str = "text_index",
        index_name: str = None,
        **kwargs,
    ):
        """
        Args:
            collection: MongoDB collection to add the texts to.
            embedding: Text embedding model to use.
            text_key: MongoDB field that will contain the text for each document.
            embedding_key: MongoDB field that will contain the embedding for each document.
            vector_index_name: Name of the Atlas Vector Search index.
            relevance_score_fn: The similarity score used for the index.
                Currently supported: 'euclidean', 'cosine', and 'dotProduct'.
            fulltext_index_name: If strategy provided to search / retriever in ["fulltext", "hybrid"],
                one must provide the name of an Atlas Search Index.
        """
        self._collection = collection
        self._embedding = embedding
        self._vector_index_name = vector_index_name
        self._text_key = text_key
        self._embedding_key = embedding_key
        self._relevance_score_fn = relevance_score_fn
        self._text_index_name = fulltext_index_name
        if index_name is not None:
            logger.warning("index_name is deprecated. Please use vector_index_name")

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
        texts_batch = texts
        metadatas_batch = _metadatas
        result_ids = []
        if batch_size:
            texts_batch = []
            metadatas_batch = []
            size = 0
            for i, (text, metadata) in enumerate(zip(texts, _metadatas)):
                size += len(text) + len(metadata)
                texts_batch.append(text)
                metadatas_batch.append(metadata)
                if (i + 1) % batch_size == 0 or size >= 47_000_000:
                    result_ids.extend(self._insert_texts(texts_batch, metadatas_batch))
                    texts_batch = []
                    metadatas_batch = []
                    size = 0
        if texts_batch:
            result_ids.extend(self._insert_texts(texts_batch, metadatas_batch))  # type: ignore
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

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        strategy: str = "vector",
        oversampling_factor: int = 10,
        include_embeddings: bool = False,
        fulltext_search_query: str = "",
        fulltext_search_operator: str = "phrase",
        vector_penalty: float = 0.0,
        fulltext_penalty: float = 0.0,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return MongoDB documents most similar to the given query and their scores.

        Extended Query interface supporting Vector, Full-Text and Hybrid Search.
        Each of the search types allows for filtering and permits a limit (k) of documents to return.

        These efficient queries require search indexes to be created in MongoDB Atlas.

        Atlas Search eliminates the need to run a separate
        search system alongside your database.

        For details on search_type='vector', see the following:
            https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/

        For details on search_type='text', see
            https://www.mongodb.com/docs/atlas/atlas-search/aggregation-stages/search/#mongodb-pipeline-pipe.-search

        For details on search_type='hybrid', see
            https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/reciprocal-rank-fusion/
            In the scoring algorithm used, Reciprocal Rank Fusion,
            scores := \frac{1}{rank + penalty} with rank in [1,2,..,n]

         Args:
            query: Input text of semantic query
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: (Optional) dictionary of argument(s) to prefilter document fields on.
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages for postprocessing.
            strategy: (Optional) Type of search to make. One of "vector", "fulltext", or "hybrid".
            oversampling_factor: Multiple of k used when generating number of candidates in HNSW Vector Search,
            include_embeddings: If True, the embedding vector of each result will be included in metadata.
            vector_penalty: Weighting factor applied to vector search results.
            fulltext_penalty: Weighting factor applied to full-text search results.
            fulltext_search_query: Input test query in the following search:
                {$search: {operator: {"query": query, "path": text_key}, "index": fulltext_index_name}}
            fulltext_search_operator: A number of operators are available in the full-text search stage.
                For details, see https://www.mongodb.com/docs/atlas/atlas-search/operators-and-collectors/#std-label-operators-ref
            kwargs: Additional arguments are specific to the search_type

        Returns:
            List of documents most similar to the query and their scores.
        """
        embedding = self._embedding.embed_query(query)
        docs = self._similarity_search_with_score(
            embedding,
            k=k,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            strategy=strategy,
            oversampling_factor=oversampling_factor,
            include_embeddings=include_embeddings,
            fulltext_search_query=fulltext_search_query,
            fulltext_search_operator=fulltext_search_operator,
            vector_penalty=vector_penalty,
            fulltext_penalty=fulltext_penalty,
            **kwargs,
        )
        return docs

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        strategy: str = "vector",
        oversampling_factor: int = 10,
        include_scores: bool = True,
        include_embeddings: bool = False,
        fulltext_search_query: str = "",
        fulltext_search_operator: str = "phrase",
        vector_penalty: float = 0.0,
        fulltext_penalty: float = 0.0,
        **kwargs: Any,
    ) -> List[Document]:
        """Return MongoDB documents most similar to the given query and their scores.

        Extended Query interface supporting Vector, Full-Text and Hybrid Search.
        Each of the search types allows for filtering and permits a limit (k) of documents to return.

        These efficient queries require search indexes to be created in MongoDB Atlas.

        Atlas Search eliminates the need to run a separate
        search system alongside your database.

        For details on search_type='vector', see the following:
            https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/

        For details on search_type='text', see
            https://www.mongodb.com/docs/atlas/atlas-search/aggregation-stages/search/#mongodb-pipeline-pipe.-search

        For details on search_type='hybrid', see
            https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/reciprocal-rank-fusion/
            In the scoring algorithm used, Reciprocal Rank Fusion,
            scores := \frac{1}{rank + penalty} with rank in [1,2,..,n]

         Args:
            query: Input text of semantic query
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: (Optional) dictionary of argument(s) to prefilter document fields on.
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages for postprocessing.
            strategy: (Optional) Type of search to make. One of "vector", "fulltext", or "hybrid".
            oversampling_factor: Multiple of k used when generating number of candidates in HNSW Vector Search,
            include_scores: If True, search scores will be included in result metadata
            include_embeddings: If True, the embedding vector of each result will be included in metadata.
            vector_penalty: Weighting factor applied to vector search results.
            fulltext_penalty: Weighting factor applied to full-text search results.
            fulltext_search_query: Input test query in the following search:
                {$search: {operator: {"query": query, "path": text_key}, "index": fulltext_index_name}}
            fulltext_search_operator: A number of operators are available in the full-text search stage.
                For details, see https://www.mongodb.com/docs/atlas/atlas-search/operators-and-collectors/#std-label-operators-ref
            kwargs: Additional arguments are specific to the search_type

        Returns:
            List of documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(
            query,
            k=k,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            strategy=strategy,
            oversampling_factor=oversampling_factor,
            include_embeddings=include_embeddings,
            fulltext_search_query=fulltext_search_query,
            fulltext_search_operator=fulltext_search_operator,
            vector_penalty=vector_penalty,
            fulltext_penalty=fulltext_penalty,
            **kwargs,
        )

        if include_scores:
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
        strategy: str = "vector",
        oversampling_factor: int = 10,
        fulltext_search_query: str = "",
        fulltext_search_operator: str = "phrase",
        vector_penalty: float = 0.0,
        fulltext_penalty: float = 0.0,
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

         Args:
            query: Input text of semantic query
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: (Optional) dictionary of argument(s) to prefilter document fields on.
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages for postprocessing.
            strategy: (Optional) Type of search to make. One of "vector", "fulltext", or "hybrid".
            oversampling_factor: Multiple of k used when generating number of candidates in HNSW Vector Search,
            vector_penalty: Weighting factor applied to vector search results.
            fulltext_penalty: Weighting factor applied to full-text search results.
            fulltext_search_query: Input test query in the following search:
                {$search: {operator: {"query": query, "path": text_key}, "index": fulltext_index_name}}
            fulltext_search_operator: A number of operators are available in the full-text search stage.
                For details, see https://www.mongodb.com/docs/atlas/atlas-search/operators-and-collectors/#std-label-operators-ref
            kwargs: Additional arguments are specific to the search_type

        Returns:
            List of documents selected by maximal marginal relevance.
        """
        return self.max_marginal_relevance_search_by_vector(
            query_vector=self._embedding.embed_query(query),
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            strategy=strategy,
            oversampling_factor=oversampling_factor,
            fulltext_search_query=fulltext_search_query,
            fulltext_search_operator=fulltext_search_operator,
            vector_penalty=vector_penalty,
            fulltext_penalty=fulltext_penalty,
            **kwargs,
        )

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
        vectorstore.add_texts(texts, metadatas=metadatas)
        return vectorstore

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by ObjectId or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        search_params: dict[str, Any] = {}
        if ids:
            search_params[self._text_key]["$in"] = ids

        return self._collection.delete_many({**search_params, **kwargs}).acknowledged

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
        query_vector: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        strategy: str = "vector",
        oversampling_factor: int = 10,
        fulltext_search_query: str = "",
        fulltext_search_operator: str = "phrase",
        vector_penalty: float = 0.0,
        fulltext_penalty: float = 0.0,
        **kwargs: Any,
    ) -> List[Document]:  # type: ignore
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query_vector: Embedding to look up documents similar to.
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
            strategy: (Optional) Type of search to make. One of "vector", "fulltext", or "hybrid".
            oversampling_factor: Multiple of k used when generating number of candidates in HNSW Vector Search,
            vector_penalty: Weighting factor applied to vector search results.
            fulltext_penalty: Weighting factor applied to full-text search results.
            fulltext_search_query: Input test query in the following search:
                {$search: {operator: {"query": query, "path": text_key}, "index": fulltext_index_name}}
            fulltext_search_operator: A number of operators are available in the full-text search stage.
                For details, see https://www.mongodb.com/docs/atlas/atlas-search/operators-and-collectors/#std-label-operators-ref
            kwargs: Additional arguments are specific to the search_type

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        docs = self._similarity_search_with_score(
            query_vector,
            k=fetch_k,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            include_embeddings=True,
            strategy=strategy,
            oversampling_factor=oversampling_factor,
            fulltext_search_query=fulltext_search_query,
            fulltext_search_operator=fulltext_search_operator,
            vector_penalty=vector_penalty,
            fulltext_penalty=fulltext_penalty,
            **kwargs,
        )
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(query_vector),
            [doc.metadata[self._embedding_key] for doc, _ in docs],
            k=k,
            lambda_mult=lambda_mult,
        )
        mmr_docs = [docs[i][0] for i in mmr_doc_indexes]
        return mmr_docs

    async def amax_marginal_relevance_search_by_vector(
        self,
        query_vector: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        strategy: str = "vector",
        oversampling_factor: int = 10,
        fulltext_search_query: str = "",
        fulltext_search_operator: str = "phrase",
        vector_penalty: float = 0.0,
        fulltext_penalty: float = 0.0,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search_by_vector,
            query_vector,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            pre_filter=pre_filter,
            post_filter_pipeline=post_filter_pipeline,
            strategy=strategy,
            oversampling_factor=oversampling_factor,
            fulltext_search_query=fulltext_search_query,
            fulltext_search_operator=fulltext_search_operator,
            vector_penalty=vector_penalty,
            fulltext_penalty=fulltext_penalty,
            **kwargs,
        )

    def _similarity_search_with_score(
        self,
        query_vector: List[float],
        k: int = 4,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        strategy: str = "vector",
        oversampling_factor: int = 10,
        include_embeddings: bool = False,
        fulltext_search_query: str = "",
        fulltext_search_operator: str = "phrase",
        vector_penalty: float = 0.0,
        fulltext_penalty: float = 0.0,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Extended Query interface supporting Vector, Full-Text and Hybrid Search.

        Each of the search types allows for filtering and permits a limit (k) of documents to return.

        These efficient queries require search indexes to be created in MongoDB Atlas.

        Built on Apache Lucene, Atlas Search eliminates the need to run a separate
        search system alongside your database.

        For details on search_type='vector', see the following:
            https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/

        For details on search_type='text', see
            https://www.mongodb.com/docs/atlas/atlas-search/aggregation-stages/search/#mongodb-pipeline-pipe.-search

        For details on search_type='hybrid', see
            https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/reciprocal-rank-fusion/
            In the scoring algorithm used, Reciprocal Rank Fusion,
            scores := \frac{1}{rank + penalty} with rank in [1,2,..,n]

         Args:
            query_vector: Embedding vector of semantic query
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: (Optional) dictionary of argument(s) to prefilter document fields on.
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages for postprocessing.
            strategy: (Optional) Type of search to make. One of "vector", "fulltext", or "hybrid".
            oversampling_factor: Multiple of k used when generating number of candidates in HNSW Vector Search,
            include_embeddings: If True, the embedding vector of each result will be included in metadata.
            vector_penalty: Weighting factor applied to vector search results.
            fulltext_penalty: Weighting factor applied to full-text search results.
            fulltext_search_query: Input test query in the following search:
                {$search: {operator: {"query": query, "path": text_key}, "index": fulltext_index_name}}
            fulltext_search_operator: A number of operators are available in the full-text search stage.
                For details, see https://www.mongodb.com/docs/atlas/atlas-search/operators-and-collectors/#std-label-operators-ref
            kwargs: Additional arguments are specific to the search_type

        Returns:
            List of documents most similar to the query and their scores.
        """

        pipeline = []

        if strategy == "vector":
            # Atlas Vector Search, potentially with filter
            pipeline = [
                vector_search_stage(
                    query_vector,
                    self._embedding_key,
                    self._vector_index_name,
                    k,
                    pre_filter,
                    oversampling_factor,
                    **kwargs,
                ),
                {"$set": {"score": {"$meta": "vectorSearchScore"}}},
            ]

        elif strategy == "fulltext":
            # Atlas Full-Text Search, potentially with filter
            pipeline = [
                text_search_stage(
                    fulltext_search_query,
                    self._text_key,
                    self._text_index_name,
                    fulltext_search_operator,
                    **kwargs,
                ),
                {"$match": {"$and": pre_filter} if pre_filter else {}},
                {"$set": {"score": {"$meta": "searchScore"}}},
                {"$limit": k},
            ]

        elif strategy == "hybrid":
            # Combines Vector and Full-Text searches with Reciprocal Rank Fusion weighting
            scores_fields = ["vector_score", "fulltext_score"]
            # Vector Search pipeline
            vector_pipeline = [
                vector_search_stage(
                    query_vector,
                    self._embedding_key,
                    self._vector_index_name,
                    k,
                    pre_filter,
                    oversampling_factor,
                    **kwargs,
                )
            ]
            vector_pipeline.extend(
                reciprocal_rank_stage(self._text_key, "vector_score", vector_penalty)
            )
            combine_pipelines(pipeline, vector_pipeline, self._collection.name)

            # Full-Text Search pipeline
            if fulltext_search_query:
                text_pipeline = [
                    text_search_stage(
                        fulltext_search_query,
                        self._text_key,
                        self._text_index_name,
                        fulltext_search_operator,
                    ),
                    {"$match": {"$and": pre_filter} if pre_filter else {}},
                    {"$limit": k},
                ]
                text_pipeline.extend(
                    reciprocal_rank_stage(
                        self._text_key, "fulltext_score", fulltext_penalty
                    )
                )
                combine_pipelines(pipeline, text_pipeline, self._collection.name)
            else:
                logger.warning(
                    f"{strategy=} selected but fulltext_search_query is empty. "
                    f"No text search will be performed, but scores are still RRF."
                )

            # Sum and sort pipeline
            pipeline += final_hybrid_stage(
                scores_fields=scores_fields, limit=k, **kwargs
            )

        else:
            raise ValueError(
                f"Unrecognized {strategy=}. Expecting one of [vector, fulltext, hybrid]"
            )

        # Remove embeddings unless requested.
        if not include_embeddings:
            pipeline.append({"$project": {self._embedding_key: 0}})
        # Post-processing
        if post_filter_pipeline is not None:
            pipeline.extend(post_filter_pipeline)

        # Execution
        cursor = self._collection.aggregate(pipeline)  # type: ignore[arg-type]
        docs = []

        # Format
        for res in cursor:
            text = res.pop(self._text_key)
            score = res.pop("score")
            make_serializable(res)
            docs.append((Document(page_content=text, metadata=res), score))
        return docs
