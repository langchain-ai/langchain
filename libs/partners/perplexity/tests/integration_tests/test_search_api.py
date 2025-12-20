"""Integration tests for Perplexity Search API."""

import os

import pytest
from langchain_core.documents import Document

from langchain_perplexity import PerplexitySearchResults, PerplexitySearchRetriever


@pytest.mark.skipif(not os.environ.get("PPLX_API_KEY"), reason="PPLX_API_KEY not set")
class TestPerplexitySearchAPI:
    def test_search_retriever_basic(self) -> None:
        """Test basic search with retriever."""
        retriever = PerplexitySearchRetriever(k=3)
        docs = retriever.invoke("What is the capital of France?")
        assert len(docs) > 0
        assert isinstance(docs[0], Document)
        assert "Paris" in docs[0].page_content
        assert docs[0].metadata["title"]
        assert docs[0].metadata["url"]

    def test_search_retriever_with_filters(self) -> None:
        """Test search with filters."""
        # Search for recent news (recency filter)
        retriever = PerplexitySearchRetriever(
            k=3, search_recency_filter="month", search_domain_filter=["wikipedia.org"]
        )
        docs = retriever.invoke("Python programming language")
        assert len(docs) > 0
        for doc in docs:
            assert "wikipedia.org" in doc.metadata["url"]

    def test_search_tool_basic(self) -> None:
        """Test basic search with tool."""
        tool = PerplexitySearchResults(max_results=3)
        results = tool.invoke("Who won the 2024 Super Bowl?")

        # BaseTool.invoke calls _run. If return_direct is False (default),
        # it returns the output of _run, which is a list of dicts.
        assert isinstance(results, list)
        assert len(results) > 0
        assert "title" in results[0]
        assert "url" in results[0]
        assert "snippet" in results[0]

    def test_search_tool_multi_query(self) -> None:
        """Test search tool with multiple queries."""
        tool = PerplexitySearchResults(max_results=2)
        queries = ["Apple stock price", "Microsoft stock price"]
        # Pass input as dict to avoid BaseTool validation error with list
        results = tool.invoke({"query": queries})

        assert isinstance(results, list)
        # Should have results for both (combined)
        assert len(results) > 0
