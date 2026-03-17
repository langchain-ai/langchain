"""Tests for TTTPoTTool."""
import pytest
from unittest.mock import patch, MagicMock
from ttt_pot_tool import TTTPoTTool, TTTPoTVerifyTool


def test_tool_name():
    tool = TTTPoTTool()
    assert tool.name == "ttt_pot_generate"


def test_verify_tool_name():
    tool = TTTPoTVerifyTool()
    assert tool.name == "ttt_pot_verify"


@patch("httpx.Client")
def test_generate_pot(mock_client_cls):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"potHash": "0xabc", "status": "ok"}
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    tool = TTTPoTTool()
    result = tool._run(tx_hash="0x123")
    assert result["status"] == "ok"
