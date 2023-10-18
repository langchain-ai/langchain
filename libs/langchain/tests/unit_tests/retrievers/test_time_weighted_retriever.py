"""Tests for the time-weighted retriever class."""

from datetime import datetime, timedelta
from typing import Any, Iterable, List, Optional, Tuple, Type

import pytest

from langchain.retrievers.time_weighted_retriever import (
    TimeWeightedVectorStoreRetriever,
    _get_hours_passed,
)
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore


def _get_example_memories(k: int = 4) -> List[Document]:
    return [
        Document(
            page_content="foo",
            metadata={
                "buffer_idx": i,
                "last_accessed_at": datetime(2023, 4, 14, 12, 0),
            },
        )
        for i in range(k)
    ]


class MockVectorStore(VectorStore):
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
        return list(texts)

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
        cls: Type["MockVectorStore"],
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "MockVectorStore":
        """Return VectorStore initialized from documents and embeddings."""
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)

    @classmethod
    def from_texts(
        cls: Type["MockVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "MockVectorStore":
        """Return VectorStore initialized from texts and embeddings."""
        return cls()

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and similarity scores, normalized on a scale from 0 to 1.

        0 is dissimilar, 1 is most similar.
        """
        return [(doc, 0.5) for doc in _get_example_memories()]


@pytest.fixture
def time_weighted_retriever() -> TimeWeightedVectorStoreRetriever:
    vectorstore = MockVectorStore()
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, memory_stream=_get_example_memories()
    )


def test__get_hours_passed() -> None:
    time1 = datetime(2023, 4, 14, 14, 30)
    time2 = datetime(2023, 4, 14, 12, 0)
    expected_hours_passed = 2.5
    hours_passed = _get_hours_passed(time1, time2)
    assert hours_passed == expected_hours_passed


def test_get_combined_score(
    time_weighted_retriever: TimeWeightedVectorStoreRetriever,
) -> None:
    document = Document(
        page_content="Test document",
        metadata={"last_accessed_at": datetime(2023, 4, 14, 12, 0)},
    )
    vector_salience = 0.7
    expected_hours_passed = 2.5
    current_time = datetime(2023, 4, 14, 14, 30)
    combined_score = time_weighted_retriever._get_combined_score(
        document, vector_salience, current_time
    )
    expected_score = (
        1.0 - time_weighted_retriever.decay_rate
    ) ** expected_hours_passed + vector_salience
    assert combined_score == pytest.approx(expected_score)


def test_get_salient_docs(
    time_weighted_retriever: TimeWeightedVectorStoreRetriever,
) -> None:
    query = "Test query"
    docs_and_scores = time_weighted_retriever.get_salient_docs(query)
    want = [(doc, 0.5) for doc in _get_example_memories()]
    assert isinstance(docs_and_scores, dict)
    assert len(docs_and_scores) == len(want)
    for k, doc in docs_and_scores.items():
        assert doc in want


def test_get_relevant_documents(
    time_weighted_retriever: TimeWeightedVectorStoreRetriever,
) -> None:
    query = "Test query"
    relevant_documents = time_weighted_retriever.get_relevant_documents(query)
    want = [(doc, 0.5) for doc in _get_example_memories()]
    assert isinstance(relevant_documents, list)
    assert len(relevant_documents) == len(want)
    now = datetime.now()
    for doc in relevant_documents:
        # assert that the last_accessed_at is close to now.
        assert now - timedelta(hours=1) < doc.metadata["last_accessed_at"] <= now

    # assert that the last_accessed_at in the memory stream is updated.
    for d in time_weighted_retriever.memory_stream:
        assert now - timedelta(hours=1) < d.metadata["last_accessed_at"] <= now


def test_add_documents(
    time_weighted_retriever: TimeWeightedVectorStoreRetriever,
) -> None:
    documents = [Document(page_content="test_add_documents document")]
    added_documents = time_weighted_retriever.add_documents(documents)
    assert isinstance(added_documents, list)
    assert len(added_documents) == 1
    assert (
        time_weighted_retriever.memory_stream[-1].page_content
        == documents[0].page_content
    )
