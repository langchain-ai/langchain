"""Integration tests for AgentDBRetriever (require live API key)."""

import pytest

from langchain_agentdb import AgentDBRetriever


@pytest.mark.requires("AGENTDB_API_KEY")
def test_get_latest_documents() -> None:
    """Fetch the 3 latest items from the live API."""
    retriever = AgentDBRetriever(mode="latest", k=3)
    docs = retriever.invoke("")
    assert len(docs) > 0
    assert docs[0].metadata["title"]
    assert docs[0].page_content


@pytest.mark.requires("AGENTDB_API_KEY")
def test_search_documents() -> None:
    """Run a semantic search against the live API (requires Pro key)."""
    retriever = AgentDBRetriever(mode="search", k=3)
    docs = retriever.invoke("artificial intelligence")
    assert len(docs) > 0
    assert docs[0].metadata["confidence"] >= 0.0
