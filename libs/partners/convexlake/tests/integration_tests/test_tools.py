"""Integration tests — require CONVEXLAKE_API_KEY in the environment."""

import os

import pytest

from langchain_convexlake import ConvexLakeCatalog, ConvexLakeTrades, get_tools

_API_KEY = os.environ.get("CONVEXLAKE_API_KEY", "")
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def api_key() -> str:
    if not _API_KEY:
        pytest.skip("Set CONVEXLAKE_API_KEY to run integration tests.")
    return _API_KEY


def test_catalog_root(api_key: str) -> None:
    tool = ConvexLakeCatalog(api_key=api_key)
    result = tool.invoke({})
    assert isinstance(result, dict)
    assert len(result) > 0


def test_catalog_drill_down(api_key: str) -> None:
    tool = ConvexLakeCatalog(api_key=api_key)
    result = tool.invoke({"exchange": "polymarket"})
    assert isinstance(result, dict)


def test_get_tools_returns_seven(api_key: str) -> None:
    tools = get_tools(api_key=api_key)
    assert len(tools) == 7
    names = {t.name for t in tools}
    assert "convexlake_catalog" in names
    assert "convexlake_trades" in names
