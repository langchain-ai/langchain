"""Integration tests for Bocha Search tools."""

import os

import pytest

from langchain_bocha import BochaSearchResults, BochaSearchRun


@pytest.mark.skipif(
    not os.environ.get("BOCHA_API_KEY"),
    reason="BOCHA_API_KEY environment variable not set",
)
def test_bocha_search_run() -> None:
    """Test BochaSearchRun tool with a live query."""
    tool = BochaSearchRun()
    output = tool.run({"query": "北京天气"})
    assert isinstance(output, str)
    assert len(output) > 0
    assert "Error:" not in output


@pytest.mark.skipif(
    not os.environ.get("BOCHA_API_KEY"),
    reason="BOCHA_API_KEY environment variable not set",
)
def test_bocha_search_results() -> None:
    """Test BochaSearchResults tool with a live query."""
    tool = BochaSearchResults()
    results = tool.run({"query": "北京天气"})
    assert isinstance(results, list)
    assert len(results) > 0
    for res in results:
        assert "title" in res
        assert "link" in res
        assert "snippet" in res


@pytest.mark.skipif(
    not os.environ.get("BOCHA_API_KEY"),
    reason="BOCHA_API_KEY environment variable not set",
)
@pytest.mark.asyncio
async def test_bocha_search_run_async() -> None:
    """Test BochaSearchRun tool asynchronously with a live query."""
    tool = BochaSearchRun()
    output = await tool.ainvoke({"query": "北京天气"})
    assert isinstance(output, str)
    assert len(output) > 0
    assert "Error:" not in output


@pytest.mark.skipif(
    not os.environ.get("BOCHA_API_KEY"),
    reason="BOCHA_API_KEY environment variable not set",
)
@pytest.mark.asyncio
async def test_bocha_search_results_async() -> None:
    """Test BochaSearchResults tool asynchronously with a live query."""
    tool = BochaSearchResults()
    results = await tool.ainvoke({"query": "北京天气"})
    assert isinstance(results, list)
    assert len(results) > 0
