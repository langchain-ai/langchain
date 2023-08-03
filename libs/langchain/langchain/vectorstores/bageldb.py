"""BagelDB integration"""
from __future__ import annotations

import uuid
from typing import (
    Optional,
    List,
    Any,
    Dict,
    Callable,
    Iterable,
    Type,
    Tuple,
)

from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.utils import xor_args

import bagel
import bagel.config
# from bagel.api.types import ID, OneOrMany

DEFAULT_K = 5


def _results_to_docs(results: Any) -> List[Document]:
    return [doc for doc, _ in _results_to_docs_and_scores(results)]


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    return [(
        Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


class Bagel(VectorStore):
    _LANGCHAIN_DEFAULT_CLUSTER_NAME = "langchain"

    def __init__(
        self,
        cluster_name: str = _LANGCHAIN_DEFAULT_CLUSTER_NAME,
        client_settings: Optional[bagel.config.Settings] = None,
        cluster_metadata: Optional[Dict] = None,
        client: Optional[bagel.Client] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        """Initialize with bagel client."""
        if client is not None:
            self._client_settings = client_settings
            self._client = client
        else:
            if client_settings:
                _client_settings = client_settings
            else:
                _client_settings = bagel.config.Settings(
                        bagel_api_impl="rest",
                        bagel_server_host="api.bageldb.ai",
                    )
            self._client_settings = _client_settings
            self._client = bagel.Client(_client_settings)

        self._cluster = self._client.get_or_create_cluster(
            name=cluster_name,
            metadata=cluster_metadata,
        )
        self.override_relevance_score_fn = relevance_score_fn

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return None

    @xor_args(("query_texts", "query_embeddings"))
    def __query_cluster(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 4,
        where: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Query bagel dataset."""
        return self._cluster.find(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            **kwargs,
        )

    def add_texts(
        self,
        texts: Iterable[str],
        embeddings: Optional[List[float]] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        # creating unique ids if None
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        texts = list(texts)

        if metadatas:
            length_diff = len(texts) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
            empty_ids = []
            non_empty_ids = []
            for idx, metadata in enumerate(metadatas):
                if metadata:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)
            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                texts_with_metadatas = [texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                self._cluster.upsert(
                    metadatas=metadatas,
                    embeddings=embeddings_with_metadatas,
                    documents=texts_with_metadatas,
                    ids=ids_with_metadata,
                )
            if empty_ids:
                texts_without_metadatas = [texts[j] for j in empty_ids]
                embeddings_without_metadatas = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self._cluster.upsert(
                    embeddings=embeddings_without_metadatas,
                    documents=texts_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            metadatas = [{}] * len(texts)
            self._cluster.upsert(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        where: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(
            query, k, where=where
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        where: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        results = self.__query_cluster(
            query_texts=[query], n_results=k, where=where
        )
        return _results_to_docs_and_scores(results)

    @classmethod
    def from_texts(
        cls: Type[Bagel],
        texts: List[str],
        cluster_name: str = _LANGCHAIN_DEFAULT_CLUSTER_NAME,
        client_settings: Optional[bagel.config.Settings] = None,
        cluster_metadata: Optional[Dict] = None,
        embeddings: Optional[List[float]] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        client: Optional[bagel.Client] = None,
        **kwargs: Any,
    ) -> Bagel:
        bagel_cluster = cls(
            cluster_name=cluster_name,
            client_settings=client_settings,
            client=client,
            cluster_metadata=cluster_metadata,
            **kwargs,
        )
        bagel_cluster.add_texts(
            texts=texts, embeddings=embeddings,
            metadatas=metadatas, ids=ids
        )
        return bagel_cluster

    def delete_cluster(self) -> None:
        """Delete the collection."""
        self._client.delete_cluster(self._cluster.name)

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        where: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        results = self.__query_cluster(
            query_embeddings=embedding, n_results=k, where=where
        )
        return _results_to_docs_and_scores(results)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        where: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        results = self.__query_cluster(
            query_embeddings=embedding, n_results=k, where=where
        )
        return _results_to_docs(results)

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self.override_relevance_score_fn:
            return self.override_relevance_score_fn

        distance = "l2"
        distance_key = "hnsw:space"
        metadata = self._cluster.metadata

        if metadata and distance_key in metadata:
            distance = metadata[distance_key]

        if distance == "cosine":
            return self._cosine_relevance_score_fn
        elif distance == "l2":
            return self._euclidean_relevance_score_fn
        elif distance == "ip":
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance metric of type: {distance}."
                "Consider providing relevance_score_fn to Chroma constructor."
            )
