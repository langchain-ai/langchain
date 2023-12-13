import pytest

from langchain_community.tools.ddg_search.tool import (
    DuckDuckGoSearchResults,
    DuckDuckGoSearchRun,
)


def ddg_installed() -> bool:
    try:
        from duckduckgo_search import DDGS  # noqa: F401

        return True
    except Exception as e:
        print(f"duckduckgo not installed, skipping test {e}")
        return False


@pytest.mark.skipif(not ddg_installed(), reason="requires duckduckgo-search package")
def test_ddg_search_tool() -> None:
    keywords = "Bella Ciao"
    tool = DuckDuckGoSearchRun()
    result = tool(keywords)
    print(result)
    assert len(result.split()) > 20


@pytest.mark.skipif(not ddg_installed(), reason="requires duckduckgo-search package")
def test_ddg_search_news_tool() -> None:
    keywords = "Tesla"
    tool = DuckDuckGoSearchResults(source="news")
    result = tool(keywords)
    print(result)
    assert len(result.split()) > 20
