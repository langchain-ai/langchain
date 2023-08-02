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
from bagel.api.types import ID, Where

DEFAULT_K = 5


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
        collection_name: str = _LANGCHAIN_DEFAULT_CLUSTER_NAME,
        client_settings: Optional[bagel.config.Settings] = None,
        collection_metadata: Optional[Dict] = None,
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
            name=collection_name,
            metadata=collection_metadata,
        )
        self.override_relevance_score_fn = relevance_score_fn

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return None

    @xor_args(("query_texts", "query_embeddings"))
    def __query_collection(
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
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        # creating unique ids if None
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        embeddings = None
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
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(
            query, k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        results = self.__query_collection(
            query_texts=[query], n_results=k, where=filter
        )
        return _results_to_docs_and_scores(results)

    @classmethod
    def from_texts(
        cls: Type[Bagel],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_CLUSTER_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[bagel.config.Settings] = None,
        client: Optional[bagel.Client] = None,
        collection_metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Bagel:
        pass
