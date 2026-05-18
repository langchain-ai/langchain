"""Unit tests for langchain-mercatai tools."""

import json
from unittest.mock import MagicMock, patch

import pytest

from langchain_mercatai import (
    MercataiDeliverTool,
    MercataiJobFetchTool,
    MercataiSubmitBidTool,
)


def _mock_client(tasks=None, bid=None, delivery=None):
    client = MagicMock()
    client.list_tasks.return_value = tasks or [{"id": "t1", "title": "Test", "budget_max_eur": 100}]
    client.bid.return_value = bid or {"id": "b1", "score": 0.85}
    client.deliver.return_value = delivery or {"id": "t1", "status": "review"}
    return client


class TestMercataiJobFetchTool:
    def test_returns_json_string(self):
        tool = MercataiJobFetchTool()
        with patch("langchain_mercatai.tools._get_client", return_value=_mock_client()):
            result = tool._run()
        data = json.loads(result)
        assert isinstance(data, list)
        assert data[0]["id"] == "t1"

    def test_passes_category(self):
        tool = MercataiJobFetchTool()
        client = _mock_client()
        with patch("langchain_mercatai.tools._get_client", return_value=client):
            tool._run(category="research", limit=5)
        client.list_tasks.assert_called_once_with(status="open", category="research", limit=5)


class TestMercataiSubmitBidTool:
    def test_submit_bid(self):
        tool = MercataiSubmitBidTool()
        with patch("langchain_mercatai.tools._get_client", return_value=_mock_client()):
            result = tool._run(task_id="t1", price_eur=80.0, estimated_hours=4.0)
        data = json.loads(result)
        assert data["id"] == "b1"


class TestMercataiDeliverTool:
    def test_deliver(self):
        tool = MercataiDeliverTool()
        with patch("langchain_mercatai.tools._get_client", return_value=_mock_client()):
            result = tool._run(task_id="t1", result="My result")
        data = json.loads(result)
        assert data["status"] == "review"


class TestToolMetadata:
    def test_tool_names(self):
        assert MercataiJobFetchTool().name == "mercatai_job_fetch"
        assert MercataiSubmitBidTool().name == "mercatai_submit_bid"
        assert MercataiDeliverTool().name == "mercatai_deliver"

    def test_tools_have_descriptions(self):
        for tool_cls in [MercataiJobFetchTool, MercataiSubmitBidTool, MercataiDeliverTool]:
            assert len(tool_cls().description) > 10
