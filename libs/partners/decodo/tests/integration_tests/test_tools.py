"""Integration tests for langchain_decodo tools.

Run with a real DECODO_API_TOKEN set:

    DECODO_API_TOKEN=your_token pytest tests/integration_tests/

All tests are skipped automatically when ``DECODO_API_TOKEN`` is not set in
the environment, so this file is safe to include in CI without secrets.
"""

from __future__ import annotations

import json
import os

import pytest

# Skip the entire module if no token is configured.
pytestmark = pytest.mark.skipif(
    not os.environ.get("DECODO_API_TOKEN"),
    reason="DECODO_API_TOKEN environment variable is not set",
)


@pytest.fixture()
def scrape_tool():  # type: ignore[return]
    from langchain_decodo.tools import DecodoWebScrapeTool

    return DecodoWebScrapeTool()


@pytest.fixture()
def search_tool():  # type: ignore[return]
    from langchain_decodo.tools import DecodoSearchTool

    return DecodoSearchTool()


@pytest.fixture()
def loader():  # type: ignore[return]
    from langchain_decodo import DecodoLoader

    return DecodoLoader(urls=["https://example.com"])


# ---------------------------------------------------------------------------
# DecodoWebScrapeTool integration tests
# ---------------------------------------------------------------------------


class TestDecodoWebScrapeToolIntegration:
    def test_scrape_example_dot_com(self, scrape_tool: object) -> None:
        from langchain_decodo.tools import DecodoWebScrapeTool

        assert isinstance(scrape_tool, DecodoWebScrapeTool)
        result = scrape_tool._run("https://example.com")  # type: ignore[union-attr]
        assert isinstance(result, str)
        assert len(result) > 0
        # example.com always contains "Example Domain"
        assert "Example Domain" in result or "example" in result.lower()

    def test_scrape_returns_string(self, scrape_tool: object) -> None:
        from langchain_decodo.tools import DecodoWebScrapeTool

        assert isinstance(scrape_tool, DecodoWebScrapeTool)
        result = scrape_tool._run("https://httpbin.org/html")  # type: ignore[union-attr]
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# DecodoSearchTool integration tests
# ---------------------------------------------------------------------------


class TestDecodoSearchToolIntegration:
    def test_google_search_returns_results(self, search_tool: object) -> None:
        from langchain_decodo.tools import DecodoSearchTool

        assert isinstance(search_tool, DecodoSearchTool)
        raw = search_tool._run("Python programming language", engine="google", num_results=3)  # type: ignore[union-attr]
        results = json.loads(raw)
        assert isinstance(results, list)
        assert len(results) > 0
        # Each result should have a content field
        for item in results:
            assert "content" in item

    def test_amazon_search_returns_results(self, search_tool: object) -> None:
        from langchain_decodo.tools import DecodoSearchTool

        assert isinstance(search_tool, DecodoSearchTool)
        raw = search_tool._run("Python book", engine="amazon", num_results=3)  # type: ignore[union-attr]
        results = json.loads(raw)
        assert isinstance(results, list)

    def test_reddit_search_prepends_filter(self, search_tool: object) -> None:
        from langchain_decodo.tools import DecodoSearchTool

        assert isinstance(search_tool, DecodoSearchTool)
        # Should not raise; results may be empty depending on query
        raw = search_tool._run("best Python libraries", engine="reddit", num_results=3)  # type: ignore[union-attr]
        results = json.loads(raw)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# DecodoLoader integration tests
# ---------------------------------------------------------------------------


class TestDecodoLoaderIntegration:
    def test_load_single_url(self) -> None:
        from langchain_core.documents import Document

        from langchain_decodo import DecodoLoader

        loader = DecodoLoader(urls="https://example.com")
        docs = loader.load()
        assert isinstance(docs, list)
        assert len(docs) > 0
        doc = docs[0]
        assert isinstance(doc, Document)
        assert doc.metadata.get("url") == "https://example.com"
        assert doc.metadata.get("source") == "https://example.com"

    def test_lazy_load_yields_documents(self) -> None:
        from langchain_core.documents import Document

        from langchain_decodo import DecodoLoader

        loader = DecodoLoader(urls=["https://example.com"])
        docs = list(loader.lazy_load())
        assert all(isinstance(d, Document) for d in docs)

    def test_continue_on_error_skips_bad_url(self) -> None:
        from langchain_decodo import DecodoLoader

        loader = DecodoLoader(
            urls=["https://this-domain-definitely-does-not-exist-xyz123.com"],
            continue_on_error=True,
        )
        docs = loader.load()
        # Should not raise; yields empty doc with error metadata
        assert isinstance(docs, list)
        if docs:
            assert "error" in docs[0].metadata
