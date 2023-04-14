"""Tests for the time-weighted retriever class."""

from typing import Any, Iterable, List, Optional, Type

import pytest
from pydantic import ValidationError

from langchain.embeddings.base import Embeddings
from langchain.retrievers.time_weighted_retriever import (
    TimeWeightedVectorStoreRetriever,
)
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore


class MockInvalidVectorStore(VectorStore):
    """Mock invalid vector store."""

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
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        return []

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore."""
        raise NotImplementedError

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        return []

    @classmethod
    def from_documents(
        cls: Type["MockInvalidVectorStore"],
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "MockInvalidVectorStore":
        """Return VectorStore initialized from documents and embeddings."""
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)

    @classmethod
    def from_texts(
        cls: Type["MockInvalidVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "MockInvalidVectorStore":
        """Return VectorStore initialized from texts and embeddings."""
        return cls()


def test_time_weighted_retriever_on_invalid_vector_store() -> None:
    vectorstore = MockInvalidVectorStore()
    with pytest.raises(
        ValidationError,
        match="Required method 'similarity_search_with_score not implemented",
    ):
        TimeWeightedVectorStoreRetriever(vectorstore=vectorstore)
