from unittest.mock import MagicMock, patch

from langchain_community.tools.disvr import DisvrDiscoverTool, DisvrToolDetails


def test_discover_tool_metadata():
    tool = DisvrDiscoverTool(api_key="dsvr_test123")
    assert tool.name == "disvr_discover"
    assert "tool" in tool.description.lower()


def test_details_tool_metadata():
    tool = DisvrToolDetails(api_key="dsvr_test123")
    assert tool.name == "disvr_tool_details"
    assert "detail" in tool.description.lower()
