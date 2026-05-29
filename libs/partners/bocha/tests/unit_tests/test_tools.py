"""Unit tests for Bocha Search tools."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_bocha import BochaSearchResults, BochaSearchRun
from langchain_bocha._client import BochaClient

MOCK_RESPONSE = {
    "code": 200,
    "log_id": "abc123",
    "msg": None,
    "data": {
        "_type": "SearchResponse",
        "queryContext": {"originalQuery": "test query"},
        "webPages": {
            "webSearchUrl": "https://bochaai.com/search?q=test+query",
            "totalEstimatedMatches": 100,
            "value": [
                {
                    "id": "https://api.bochaai.com/v1/#WebPages.0",
                    "name": "Test Title 1",
                    "url": "http://example.com/1",
                    "displayUrl": "http://example.com/1",
                    "snippet": "Snippet 1",
                    "summary": "Summary 1",
                    "images": [
                        "http://example.com/img1.jpg",
                    ],
                    "siteName": "Site 1",
                    "siteIcon": "http://example.com/icon1.png",
                    "datePublished": "2026-05-28T10:00:00+08:00",
                    "dateLastCrawled": "2026-05-28T10:00:00Z",
                    "cachedPageUrl": None,
                    "language": None,
                    "isFamilyFriendly": None,
                    "isNavigational": None,
                },
                {
                    "id": "https://api.bochaai.com/v1/#WebPages.1",
                    "name": "Test Title 2",
                    "url": "http://example.com/2",
                    "displayUrl": "http://example.com/2",
                    "snippet": "Snippet 2",
                    "summary": "",
                    "images": [],
                    "siteName": "Site 2",
                    "siteIcon": "http://example.com/icon2.png",
                    "datePublished": "2026-05-27T08:00:00+08:00",
                    "dateLastCrawled": "2026-05-27T08:00:00Z",
                    "cachedPageUrl": None,
                    "language": None,
                    "isFamilyFriendly": None,
                    "isNavigational": None,
                },
            ],
            "someResultsRemoved": False,
        },
        "images": None,
        "videos": None,
    },
}


def test_bocha_search_run_tool() -> None:
    """Test invoking the BochaSearchRun tool."""
    tool = BochaSearchRun(bocha_api_key="fake")  # type: ignore[arg-type]
    tool.client = MagicMock(spec=BochaClient)
    tool.client.post.return_value = MOCK_RESPONSE

    output = tool.run({"query": "test query"})

    assert "Test Title 1" in output
    assert "Test Title 2" in output
    tool.client.post.assert_called_once()


def test_bocha_search_results_tool() -> None:
    """Test invoking the BochaSearchResults tool."""
    tool = BochaSearchResults(bocha_api_key="fake")  # type: ignore[arg-type]
    tool.client = MagicMock(spec=BochaClient)
    tool.client.post.return_value = MOCK_RESPONSE

    results = tool.run({"query": "test query"})

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]["id"] == "https://api.bochaai.com/v1/#WebPages.0"
    assert results[0]["title"] == "Test Title 1"
    assert results[0]["link"] == "http://example.com/1"
    assert results[0]["display_url"] == "http://example.com/1"
    assert results[0]["snippet"] == "Snippet 1"
    assert results[0]["summary"] == "Summary 1"
    assert results[0]["site_name"] == "Site 1"
    assert results[0]["site_icon"] == "http://example.com/icon1.png"
    assert results[0]["date_published"] == "2026-05-28T10:00:00+08:00"
    assert results[0]["date_last_crawled"] == "2026-05-28T10:00:00Z"
    assert results[0]["images"] == ["http://example.com/img1.jpg"]

    assert results[1]["title"] == "Test Title 2"
    assert results[1]["summary"] == ""
    assert results[1]["images"] == []


@pytest.mark.asyncio
async def test_bocha_search_run_tool_async() -> None:
    """Test invoking the BochaSearchRun tool asynchronously."""
    tool = BochaSearchRun(bocha_api_key="fake")  # type: ignore[arg-type]
    mock_client = MagicMock(spec=BochaClient)
    mock_client.apost = AsyncMock(return_value=MOCK_RESPONSE)
    tool.client = mock_client

    output = await tool.arun({"query": "test query"})

    assert "Test Title 1" in output
    assert "Test Title 2" in output
    mock_client.apost.assert_called_once()


@pytest.mark.asyncio
async def test_bocha_search_results_tool_async() -> None:
    """Test invoking the BochaSearchResults tool asynchronously."""
    tool = BochaSearchResults(bocha_api_key="fake")  # type: ignore[arg-type]
    mock_client = MagicMock(spec=BochaClient)
    mock_client.apost = AsyncMock(return_value=MOCK_RESPONSE)
    tool.client = mock_client

    results = await tool.arun({"query": "test query"})

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]["title"] == "Test Title 1"


def test_bocha_search_run_formatted_output() -> None:
    """Test that BochaSearchRun formats output correctly."""
    tool = BochaSearchRun(bocha_api_key="fake")  # type: ignore[arg-type]
    tool.client = MagicMock(spec=BochaClient)
    tool.client.post.return_value = MOCK_RESPONSE

    output = tool.run({"query": "test query"})

    expected = (
        "Title: Test Title 1\nLink: http://example.com/1\nSnippet: Summary 1\n\n"
        "Title: Test Title 2\nLink: http://example.com/2\nSnippet: Snippet 2\n"
    )
    assert output == expected
