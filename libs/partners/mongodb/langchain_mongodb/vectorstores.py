""" NEW ADDITIONS BEGIN AT LINE 500!! """

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
from bson import ObjectId, json_util
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.driver_info import DriverInfo

from langchain_mongodb.utils import maximal_marginal_relevance

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
        index_name: str = "vector_index",
        text_key: str = "text",
        embedding_key: str = "embedding",
        relevance_score_fn: str = "cosine",
        strategy: str = "vector",
        text_index_name: str = "text",
        text_search_operator: str = "phrase",
        text_penalty: int = 0,
        vector_penalty: int = 0,
        **kwargs,
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
            index_name: Name of the Atlas Vector Search index.
            relevance_score_fn: The similarity score used for the index.
                defaults to 'cosine'
                Currently supported: 'euclidean', 'cosine', and 'dotProduct'.
            strategy: Search index strategy to apply.  # TODO - Improve documentation
                One of 'vector', 'text', or 'hybrid'.
                defaults to 'vector'
            text_index_name: If strategy in ["text", "hybrid"], one must provide the name of an Atlas Search Index
            text_search_operator: A number of operators are available in the text search stage.
                For details, see https://www.mongodb.com/docs/atlas/atlas-search/operators-and-collectors/#std-label-operators-ref
        """
        self._collection = collection
        self._embedding = embedding
        self._index_name = index_name
        self._text_key = text_key
        self._embedding_key = embedding_key
        self._relevance_score_fn = relevance_score_fn
        self._strategy = strategy
        self._text_index_name = text_index_name
        self._text_search_operator = text_search_operator
        self.text_penalty = text_penalty
        self.vector_penalty = vector_penalty
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

    def _similarity_search_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        include_embedding: bool = False,
        **kwargs: Any,
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

        # Exclude the embedding key from the return payload
        if not include_embedding:
            pipeline.append({"$project": {self._embedding_key: 0}})

        if post_filter_pipeline is not None:
            pipeline.extend(post_filter_pipeline)
        cursor = self._collection.aggregate(pipeline)  # type: ignore[arg-type]
        docs = []

        def _make_serializable(obj: Dict[str, Any]) -> None:
            for k, v in obj.items():
                if isinstance(v, dict):
                    _make_serializable(v)
                elif isinstance(v, list) and v and isinstance(v[0], ObjectId):
                    obj[k] = [json_util.default(item) for item in v]
                elif isinstance(v, ObjectId):
                    obj[k] = json_util.default(v)

        for res in cursor:
            text = res.pop(self._text_key)
            score = res.pop("score")
            # Make every ObjectId found JSON-Serializable
            # following format used in bson.json_util.loads
            # e.g. loads('{"_id": {"$oid": "664..."}}') == {'_id': ObjectId('664..')} # noqa: E501
            _make_serializable(res)
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
        additional = kwargs.get("additional")  # TODO - I vote for removing this.
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

    def similarity_search_extended(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict] = None,
        post_filter_pipeline: Optional[List[Dict]] = None,
        # strategy: str = "vector", # TODO - Where does this belong?
        score: bool = True,
        oversampling_factor: int = 100,
        **kwargs: Any
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


         Args:
            query: Text to look up documents similar to.
            k: (Optional) number of documents to return. Defaults to 4.
            pre_filter: (Optional) dictionary of argument(s) to prefilter document
                fields on.
            post_filter_pipeline: (Optional) Pipeline of MongoDB aggregation stages
                for postprocessing).
            strategy: (Optional) One of "vector", "text", or "hybrid"
            kwargs: Additional arguments are specific to the search_type

        Returns:
            List of documents most similar to the query and their scores.
            Results contain only the

         """

        if self._strategy == "hybrid":

            pipeline = []  # combined aggregate pipeline
            query_vector = self._embedding.embed_query(query)

            # Vector Search
            vector_stage = {
                '$vectorSearch': {
                'index': self._index_name,
                'path': self._embedding_key,
                'queryVector': query_vector,
                'numCandidates': k * oversampling_factor,
                'limit': k,
                'filter': {"$and": pre_filter} if pre_filter else None,
                }
            }
            pipeline.append(vector_stage)

            # Reciprocal Rank Stage: (score_field: str, query, query_embedding, penalty, project_fields)

            pipeline.extend(self._reciprocal_rank_stage("vector_score", penalty=self.vector_penalty))

            # Full-Text Search Stage
            text_pipeline = [
                self._text_search_stage(query=query),
                {"$match": {"$and": pre_filter} if pre_filter else {}},
                {"$limit": k},
            ]
            pipeline.extend(text_pipeline)
            pipeline.extend(self._reciprocal_rank_stage("text_score", penalty=self.text_penalty))

        elif self._strategy == "text":
            text_stage = self._text_search_stage(query=query)
            pipeline = [
                text_stage,
                {"$match": {"$and": pre_filter} if pre_filter else {}},
                {"$set": {"score": {'$meta': 'searchScore'}}},
                {"$limit": k},
            ]
        elif self._strategy == "vector":
            embedding_vector = self._embedding.embed_query(query)
            pipeline = []
            raise NotImplementedError
        else:
            raise ValueError(f"{self._strategy} not understood. Expecting one of [vector, text, hybrid]")

        # with self._collection.aggregate(pipeline) as cursor:
        res = list(self._collection.aggregate(pipeline))

        return res

    def _reciprocal_rank_stage(self, score_field: str, penalty: float = 0, extra_fields: List[str] = None):
        """Pipeline stage that ranks and weights scores.

            Pushes documents retrieved from previous stage into a temporary sub-document.
            It then unwinds to establish the rank to each and applies the penalty.
        """
        rrf_pipeline = [
            {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
            {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
            {
                "$addFields": {
                    score_field: {"$divide": [1.0, {"$add": ["$rank", penalty, 1]}]}
                }
            }
        ]
        projection_fields = {self._text_key: f"$docs.{self._text_key}"}
        projection_fields["_id"] = "$docs._id"
        projection_fields[score_field] = 1
        if extra_fields:
            projection_fields.update({f"$docs.{key}" for key in extra_fields})

        rrf_pipeline.append({'$project': projection_fields})
        return rrf_pipeline


    def _vector_search_stage(
        self,
        query_vector: List[float],
        search_field: str,
        search_index_name: str,
        limit: int = 4,
        oversampling_factor=10,
        filters: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """

        A vector search's pipeline looks like this.

        pipeline = [
            vector_search_stage,
            {
                '$project': self._project_fields(
                    extra_fields={"score": {'$meta': 'vectorSearchScore'}}
                )
            },
        ]

        Within a Hybrid Search Pipeline, one would apply this stage like so.
        like so:

            vector_pipeline = [
                vector_stage,
                *self._reciprocal_rank_stage(search_field, score_field),
            ]
            self._add_stage_to_pipeline(hybrid_pipeline, vector_pipeline)

        """

        return {
            '$vectorSearch': {
                'index': search_index_name,
                'path': search_field,
                'queryVector': query_vector,
                'numCandidates': limit * oversampling_factor,
                'limit': limit,
                'filter': {"$and": filters} if filters else None,
            }
        }



    def _final_stage(self, scores_fields, limit):
        """Sum individual scores, sort, and apply limit."""
        doc_fields = self._column_infos.keys()
        grouped_fields = {
            key: {"$first": f"${key}"} for key in doc_fields if key != "_id"
        }
        best_score = {score: {'$max': f'${score}'} for score in scores_fields}
        final_pipeline = [
            {"$group": {"_id": "$_id", **grouped_fields, **best_score}},
            {
                "$project": {
                    **{doc_field: 1 for doc_field in doc_fields},
                    **{score: {"$ifNull": [f"${score}", 0]} for score in scores_fields},
                }
            },
            {
                "$addFields": {
                    "score": {"$add": [f"${score}" for score in scores_fields]},
                }
            },
            {"$sort": {"score": -1}},
            {"$limit": limit},
        ]
        return final_pipeline

    def _text_search_stage(self, query: str) -> Dict[str, Any]:
        """Full-Text search.

        query: Input text on which to search

        The Atlas Search Index name is set in the class constructor,
        as is the operator used to search index name
        The aggregation pipeline looks like so:
        """
        return {
            "$search": {
                "index": self._text_index_name,
                self._text_search_operator: {"query": query, "path": self._text_key},
            }
        }

    def _text_search(
            self,
            query: str,
            limit: int,
            search_field: str = '',
        ) -> Any:  # TODO
            """Find documents in the index based on a text search query

            :param query: The text to search for
            :param limit: maximum number of documents to return
            :param search_field: name of the field to search on
            :return: a named tuple containing `documents` and `scores`
            """
            text_stage = self._text_search_stage(query=query, search_field=search_field)

            pipeline = [
                text_stage,
                {
                    '$project': self._project_fields(
                        extra_fields={'score': {'$meta': 'searchScore'}}
                    )
                },
                {"$limit": limit},
            ]

            with self._collection.aggregate(pipeline) as cursor:
                documents, scores = self._mongo_to_docs(cursor)

            return dict(documents=documents, scores=scores)  # TODO Update type