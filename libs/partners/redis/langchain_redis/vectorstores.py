"""Redis vector store."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from redisvl.index import SearchIndex  # type: ignore[import]
from redisvl.query import RangeQuery, VectorQuery  # type: ignore[import]
from redisvl.query.filter import FilterExpression  # type: ignore[import]
from redisvl.redis.utils import buffer_to_array, convert_bytes  # type: ignore[import]

from langchain_redis.config import RedisConfig

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    try:
        import simsimd as simd  # type: ignore

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = 1 - simd.cdist(X, Y, metric="cosine")
        if isinstance(Z, float):
            return np.array([Z])
        return np.array(Z)
    except ImportError:
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: List[np.ndarray],
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance."""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


class RedisVectorStore(VectorStore):
    """Redis vector store implementation using RedisVL."""

    def __init__(
        self,
        embeddings: Embeddings,
        config: Optional[RedisConfig] = None,
        **kwargs: Any,
    ):
        self.config = config or RedisConfig(**kwargs)
        self._embeddings = embeddings

        if self.config.embedding_dimensions is None:
            self.config.embedding_dimensions = len(
                self._embeddings.embed_query(
                    "The quick brown fox jumps over the lazy dog"
                )
            )

        if self.config.index_schema:
            self._index = SearchIndex(self.config.index_schema, self.config.redis())
            self._index.create(overwrite=False)

        elif self.config.schema_path:
            self._index = SearchIndex.from_yaml(self.config.schema_path)
            self._index.set_client(self.config.redis())
            self._index.create(overwrite=False)
        elif self.config.from_existing and self.config.index_name:
            self._index = SearchIndex.from_existing(
                self.config.index_name, self.config.redis()
            )
            self._index.create(overwrite=False)
        else:
            # Set the default separator for tag fields where separator is not defined
            modified_metadata_schema = []
            if self.config.metadata_schema is not None:
                for field in self.config.metadata_schema:
                    if field["type"] == "tag":
                        if "attrs" not in field or "separator" not in field["attrs"]:
                            modified_field = field.copy()
                            modified_field.setdefault("attrs", {})["separator"] = (
                                self.config.default_tag_separator
                            )
                            modified_metadata_schema.append(modified_field)
                        else:
                            modified_metadata_schema.append(field)
                    else:
                        modified_metadata_schema.append(field)

            self._index = SearchIndex.from_dict(
                {
                    "index": {
                        "name": self.config.index_name,
                        "prefix": f"{self.config.key_prefix}",
                        "storage_type": self.config.storage_type,
                    },
                    "fields": [
                        {"name": self.config.content_field, "type": "text"},
                        {
                            "name": self.config.embedding_field,
                            "type": "vector",
                            "attrs": {
                                "dims": len(
                                    self._embeddings.embed_query(
                                        "The quick brown fox jumps over the lazy dog"
                                    )
                                ),
                                "distance_metric": self.config.distance_metric,
                                "algorithm": self.config.indexing_algorithm,
                                "datatype": self.config.vector_datatype,
                            },
                        },
                        *modified_metadata_schema,
                    ],
                }
            )
            self._index.set_client(self.config.redis())
            self._index.create(overwrite=False)

    @property
    def index(self) -> SearchIndex:
        return self._index

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    @property
    def key_prefix(self) -> Optional[str]:
        return self.config.key_prefix

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        keys: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add text documents to the vector store."""
        # Embed the documents in bulk
        # Convert texts to a list if it's not already
        texts_list = list(texts)
        embeddings = self._embeddings.embed_documents(texts_list)

        datas = [
            {
                self.config.content_field: text,
                self.config.embedding_field: np.array(
                    embedding, dtype=np.float32
                ).tobytes(),
                **{
                    field_name: (
                        self.config.default_tag_separator.join(metadata[field_name])
                        if isinstance(metadata.get(field_name), list)
                        else metadata.get(field_name)
                    )
                    for field_name in metadata
                },
            }
            for text, embedding, metadata in zip(
                texts_list, embeddings, metadatas or [{}] * len(texts_list)
            )
        ]

        result = (
            self._index.load(
                datas, keys=[f"{self.config.key_prefix}:{key}" for key in keys]
            )
            if keys
            else self._index.load(datas)
        )

        return list(result) if result is not None else []

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        config: Optional[RedisConfig] = None,
        keys: Optional[List[str]] = None,
        return_keys: bool = False,
        **kwargs: Any,
    ) -> RedisVectorStore:
        """Create a RedisVectorStore from a list of texts."""
        config = config or RedisConfig.from_kwargs(**kwargs)

        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]

        vector_store = cls(
            embeddings=embedding,
            config=config,
        )
        out_keys = vector_store.add_texts(texts, metadatas, keys)  # type: ignore

        if return_keys:
            return cast(RedisVectorStore, (vector_store, out_keys))
        else:
            return vector_store

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        config: Optional[RedisConfig] = None,
        return_keys: bool = False,
        **kwargs: Any,
    ) -> RedisVectorStore:
        """Create a RedisVectorStore from a list of texts."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        config = config or RedisConfig.from_kwargs(**kwargs)

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            config=config,
            return_keys=return_keys,
            **kwargs,
        )

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete keys from the vector store."""
        return self._index.delete(ids)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[FilterExpression] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Optional filter expression to apply.
            **kwargs: Other keyword arguments:
                - return_metadata: Whether to return metadata. Defaults to True.
                - distance_threshold: Optional distance threshold for filtering results.
                - return_all: Whether to return all data in the Hash/JSON including
                  non-indexed fields

        Returns:
            List of Documents most similar to the query vector.
        """
        return_metadata = kwargs.get("return_metadata", True)
        distance_threshold = kwargs.get("distance_threshold")
        return_all = kwargs.get("return_all", False)

        # Determine the fields to return based on the return_metadata flag
        if not return_all:
            return_fields = [self.config.content_field]
            if return_metadata:
                return_fields += [
                    field.name
                    for field in self._index.schema.fields.values()
                    if field.name
                    not in [self.config.embedding_field, self.config.content_field]
                ]
        else:
            return_fields = []

        if distance_threshold is None:
            results = self._index.query(
                VectorQuery(
                    vector=embedding,
                    vector_field_name=self.config.embedding_field,
                    return_fields=return_fields,
                    num_results=k,
                    filter_expression=filter,
                )
            )
        else:
            results = self._index.query(
                RangeQuery(
                    vector=embedding,
                    vector_field_name=self.config.embedding_field,
                    return_fields=return_fields,
                    num_results=k,
                    filter_expression=filter,
                    distance_threshold=distance_threshold,
                )
            )

        if not return_all:
            return [
                Document(
                    page_content=doc[self.config.content_field],
                    metadata=(
                        {
                            field.name: doc[field.name]
                            for field in self._index.schema.fields.values()
                            if field.name
                            not in [
                                self.config.embedding_field,
                                self.config.content_field,
                            ]
                        }
                        if return_metadata
                        else {}
                    ),
                )
                for doc in results
            ]
        else:
            # Fetch full hash data for each document
            pipe = self._index.client.pipeline()
            for doc in results:
                pipe.hgetall(doc["id"])
            full_docs = convert_bytes(pipe.execute())

            return [
                Document(
                    page_content=doc[self.config.content_field],
                    metadata={
                        k: v for k, v in doc.items() if k != self.config.content_field
                    },
                )
                for doc in full_docs
            ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[FilterExpression] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Optional filter expression to apply.
            **kwargs: Other keyword arguments to pass to the search function.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self._embeddings.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, filter, **kwargs)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[FilterExpression] = None,
        **kwargs: Any,
    ) -> Union[List[Tuple[Document, float]], List[Tuple[Document, float, np.ndarray]]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Optional filter expression to apply.
            **kwargs: Other keyword arguments:
                with_vectors: Whether to return document vectors. Defaults to False.
                return_metadata: Whether to return metadata. Defaults to True.
                distance_threshold: Optional distance threshold for filtering results.

        Returns:
            List of tuples of Documents most similar to the query vector, score, and
            optionally the document vector.
        """
        with_vectors = kwargs.get("with_vectors", False)
        return_metadata = kwargs.get("return_metadata", True)
        distance_threshold = kwargs.get("distance_threshold")
        return_all = kwargs.get("return_all", False)

        if not return_all:
            return_fields = [self.config.content_field]
            if return_metadata:
                return_fields += [
                    field.name
                    for field in self._index.schema.fields.values()
                    if field.name
                    not in [self.config.embedding_field, self.config.content_field]
                ]

            if with_vectors:
                return_fields.append(self.config.embedding_field)
        else:
            return_fields = []

        if distance_threshold is None:
            results = self._index.query(
                VectorQuery(
                    vector=embedding,
                    vector_field_name=self.config.embedding_field,
                    return_fields=return_fields,
                    num_results=k,
                    filter_expression=filter,
                )
            )
        else:
            results = self._index.query(
                RangeQuery(
                    vector=embedding,
                    vector_field_name=self.config.embedding_field,
                    return_fields=return_fields,
                    num_results=k,
                    filter_expression=filter,
                    distance_threshold=distance_threshold,
                )
            )

        if not return_all:
            if with_vectors:
                # Extract the document ids
                doc_ids = [doc["id"] for doc in results]

                # Retrieve the documents from the storage
                docs_from_storage = self._index._storage.get(
                    self._index.client, doc_ids
                )

                # Create a dictionary mapping document ids to their embeddings
                doc_embeddings_dict = {
                    doc_id: buffer_to_array(doc[self.config.embedding_field])
                    for doc_id, doc in zip(doc_ids, docs_from_storage)
                }

                # Prepare the results with embeddings
                docs_with_scores = [
                    (
                        Document(
                            page_content=doc[self.config.content_field],
                            metadata=(
                                {
                                    field.name: doc[field.name]
                                    for field in self._index.schema.fields.values()
                                    if field.name
                                    not in [
                                        self.config.embedding_field,
                                        self.config.content_field,
                                        "id",
                                    ]
                                }
                                if return_metadata
                                else {}
                            ),
                        ),
                        float(doc["vector_distance"]),
                        doc_embeddings_dict[doc[self.config.id_field]],
                    )
                    for doc in results
                ]
            else:
                # Prepare the results without embeddings
                docs_with_scores = [
                    (  # type: ignore[misc]
                        Document(
                            page_content=doc[self.config.content_field],
                            metadata=(
                                {
                                    field.name: doc[field.name]
                                    for field in self._index.schema.fields.values()
                                    if field.name
                                    not in [
                                        self.config.embedding_field,
                                        self.config.content_field,
                                        "id",
                                    ]
                                }
                                if return_metadata
                                else {}
                            ),
                        ),
                        float(doc["vector_distance"]),
                    )
                    for doc in results
                ]
        else:
            # Fetch full hash data for each document
            pipe = self._index.client.pipeline()
            for doc in results:
                pipe.hgetall(doc["id"])
            full_docs = convert_bytes(pipe.execute())

            if with_vectors:
                docs_with_scores = [
                    (
                        Document(
                            page_content=doc[self.config.content_field],
                            metadata={
                                k: v
                                for k, v in doc.items()
                                if k != self.config.content_field
                            },
                        ),
                        float(result.get("vector_distance", 0)),
                        buffer_to_array(doc.get(self.config.embedding_field)),
                    )
                    for doc, result in zip(full_docs, results)
                ]
            else:
                docs_with_scores = [
                    cast(  # type: ignore[misc]
                        Union[
                            Tuple[Document, float], Tuple[Document, float, np.ndarray]
                        ],
                        (
                            Document(
                                page_content=doc[self.config.content_field],
                                metadata={
                                    k: v
                                    for k, v in doc.items()
                                    if k != self.config.content_field
                                },
                            ),
                            float(result.get("vector_distance", 0)),
                        ),
                    )
                    for doc, result in zip(full_docs, results)
                ]

        return docs_with_scores

    def similarity_search_with_score(  # type: ignore[override]
        self,
        query: str,
        k: int = 4,
        filter: Optional[FilterExpression] = None,
        **kwargs: Any,
    ) -> Union[List[Tuple[Document, float]], List[Tuple[Document, float, np.ndarray]]]:
        """Return docs most similar to query."""
        embedding = self._embeddings.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                    Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            **kwargs: Other keyword arguments to pass to the search function.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        # Fetch top fetch_k documents based on similarity to the embedding
        docs_scores_embeddings = self.similarity_search_with_score_by_vector(
            embedding, k=fetch_k, with_vectors=True, **kwargs
        )

        # Extract documents and embeddings
        documents = []
        embeddings = []
        for item in docs_scores_embeddings:
            if len(item) == 3:
                doc, _, emb = item
                documents.append(doc)
                embeddings.append(emb)
            elif len(item) == 2:
                doc, _ = item
                documents.append(doc)

        # Perform MMR on the embeddings
        if embeddings:
            mmr_selected = maximal_marginal_relevance(
                np.array(embedding),
                embeddings,
                k=min(k, len(documents)),
                lambda_mult=lambda_mult,
            )

            # Return the selected documents based on MMR
            return [documents[i] for i in mmr_selected]
        else:
            return []

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                    Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            **kwargs: Other keyword arguments to pass to the search function.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        query_embedding = self.embeddings.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            query_embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )
