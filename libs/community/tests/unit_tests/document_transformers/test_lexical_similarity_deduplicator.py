"""Unit tests for LexicalSimilarityDeduplicator."""

import pytest
from langchain_core.documents import Document

from langchain_community.document_transformers.lexical_similarity_deduplicator import (
    LexicalSimilarityDeduplicator,
)


def test_deduplicator_removes_exact_duplicates() -> None:
    docs = [
        Document(page_content="LangChain is a framework for developing LLM apps."),
        Document(page_content="LangChain is a framework for developing LLM apps."),
        Document(page_content="Python is a popular programming language."),
    ]
    deduplicator = LexicalSimilarityDeduplicator(similarity_threshold=0.8)
    result = deduplicator.transform_documents(docs)

    assert len(result) == 2
    assert result[0].page_content == "LangChain is a framework for developing LLM apps."
    assert result[1].page_content == "Python is a popular programming language."


def test_deduplicator_removes_near_duplicates() -> None:
    docs = [
        Document(page_content="Retrieval augmented generation enhances LLMs with facts."),
        Document(page_content="Retrieval-augmented generation enhances LLM models with facts."),
        Document(page_content="Vector databases index embeddings for similarity search."),
    ]
    deduplicator = LexicalSimilarityDeduplicator(similarity_threshold=0.6, ngram_size=2)
    result = deduplicator.transform_documents(docs)

    assert len(result) == 2
    assert result[0].page_content.startswith("Retrieval")
    assert result[1].page_content.startswith("Vector")


def test_deduplicator_preserves_unique_documents() -> None:
    docs = [
        Document(page_content="Alpha beta gamma delta."),
        Document(page_content="One two three four five."),
        Document(page_content="Red blue green yellow."),
    ]
    deduplicator = LexicalSimilarityDeduplicator(similarity_threshold=0.5)
    result = deduplicator.transform_documents(docs)

    assert len(result) == 3


@pytest.mark.asyncio
async def test_async_deduplicator() -> None:
    docs = [
        Document(page_content="Deep learning models need GPUs."),
        Document(page_content="Deep learning models need GPUs."),
    ]
    deduplicator = LexicalSimilarityDeduplicator()
    result = await deduplicator.atransform_documents(docs)

    assert len(result) == 1
