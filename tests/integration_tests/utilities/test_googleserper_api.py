"""Integration test for Serper.dev's Google Search API Wrapper."""
import pytest

from langchain.utilities.google_serper import GoogleSerperAPIWrapper


def test_search_call() -> None:
    """Test that call gives the correct answer from search."""
    search = GoogleSerperAPIWrapper()
    output = search.run("What was Obama's first name?")
    assert "Barack Hussein Obama II" in output


def test_news_call() -> None:
    """Test that call gives the correct answer from news search."""
    search = GoogleSerperAPIWrapper(type="news")
    output = search.run("What's new with stock market?").lower()
    assert "stock" in output or "market" in output


async def test_results() -> None:
    """Test that call gives the correct answer."""
    search = GoogleSerperAPIWrapper()
    output = search.results("What was Obama's first name?")
    assert "Barack Hussein Obama II" in output["answerBox"]["answer"]


@pytest.mark.asyncio
async def test_async_call() -> None:
    """Test that call gives the correct answer."""
    search = GoogleSerperAPIWrapper()
    output = await search.arun("What was Obama's first name?")
    assert "Barack Hussein Obama II" in output


@pytest.mark.asyncio
async def test_async_results() -> None:
    """Test that call gives the correct answer."""
    search = GoogleSerperAPIWrapper()
    output = await search.aresults("What was Obama's first name?")
    assert "Barack Hussein Obama II" in output["answerBox"]["answer"]
