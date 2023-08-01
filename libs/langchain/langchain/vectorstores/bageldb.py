"""BagelDB integration"""
from typing import (
    Optional,
    List,
    Any,
    Dict,
    Callable,
)
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.utils import xor_args

import bagel
import bagel.config
from bagel.api.types import ID, Where


class Bagel(VectorStore):
    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

    def __init__(
        self,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        client_settings: Optional[bagel.config.Settings] = None,
        collection_metadata: Optional[Dict] = None,
        client: Optional[bagel.Client] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        """Initialize with bagel client."""
        if client is not None:
            self._client_settings = client_settings
            self._client = client
            self._persist_directory = persist_directory
        else:
            if client_settings:
                _client_settings = client_settings
            elif persist_directory:
                major, minor, _ = bagel.__version__.split(".")
                if int(major) == 0 and int(minor) < 4:
                    _client_settings = bagel.config.Settings(
                        bagel_db_impl="rest",
                    )
                else:
                    _client_settings = bagel.config.Settings(
                        is_persistent=True
                        )
                _client_settings.persist_directory = persist_directory
            else:
                _client_settings = bagel.config.Settings()
            self._client_settings = _client_settings
            self._client = bagel.Client(_client_settings)
            self._persist_directory = (
                _client_settings.persist_directory or persist_directory
            )

        self._embedding_function = embedding_function
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function.embed_documents
            if self._embedding_function is not None
            else None,
            metadata=collection_metadata,
        )
        self.override_relevance_score_fn = relevance_score_fn

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding_function

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
        return self._collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            **kwargs,
        )
