"""Unit tests for LogicNodes tool."""
from unittest.mock import MagicMock, patch
import pytest
from langchain_community.tools.logicnodes.tool import LogicNodesRegistryTool, CHAIN_ID


def test_logicnodes_tool_name():
    tool = LogicNodesRegistryTool()
    assert tool.name == "logicnodes_registry"


def test_logicnodes_tool_uses_api_key():
    tool = LogicNodesRegistryTool(api_key="test-key-123")
    assert tool.api_key == "test-key-123"


def test_logicnodes_chain_id_is_base():
    assert CHAIN_ID == 8453


@patch("langchain_community.tools.logicnodes.tool.requests.post")
def test_logicnodes_tool_run(mock_post):
    mock_response = MagicMock()
    mock_response.text = '{"registered": true, "radc": 5000000}'
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    tool = LogicNodesRegistryTool(api_key="test-key")
    result = tool._run(agent_address="0xabc123", service="reputation_lookup")

    assert "registered" in result
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "X-LogicNodes-Key" in call_kwargs.kwargs["headers"]


@patch("langchain_community.tools.logicnodes.tool.requests.post")
def test_logicnodes_tool_http_error(mock_post):
    import requests
    mock_response = MagicMock()
    mock_response.status_code = 402
    mock_response.text = "Insufficient credits"
    mock_post.return_value = mock_response
    mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )

    tool = LogicNodesRegistryTool(api_key="test-key")
    result = tool._run(agent_address="0xabc123")
    assert "error" in result.lower()
