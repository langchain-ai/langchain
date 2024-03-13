"""Test the voyageai reranker."""

from langchain.retrievers.document_compressors.voyageai_rerank import VoyageAIRerank


def test_voyageai_reranker_init() -> None:
    """Test the voyageai reranker initializes correctly."""
    VoyageAIRerank()
