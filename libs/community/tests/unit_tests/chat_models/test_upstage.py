"""Test Upstage Chat API wrapper."""
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_community.chat_model.upstage import ChatUpstage


@pytest.mark.requires("upstage")
def test_openai_model_param() -> None:
    llm = ChatUpstage(model="foo")
    assert llm.model_name == "foo"
    llm = ChatUpstage(model_name="foo")
    assert llm.model_name == "foo"


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "chatcmpl-sefo30fdj",
        "object": "chat.completion",
        "created": 1689989000,
        "model": "upstage/solar-1-mini-chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Upstage Bar",
                },
                "finish_reason": "stop",
            }
        ],
    }


@pytest.mark.requires("upstage")
def test_upstage_predict(mock_completion: dict) -> None:
    llm = ChatUpstage()
    mock_client = MagicMock()
    completed = False

    def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        res = llm.predict("bar")
        assert res == "Upstage Bar"
    assert completed


@pytest.mark.requires("upstage")
async def test_upstage_apredict(mock_completion: dict) -> None:
    llm = ChatUpstage()
    mock_client = MagicMock()
    completed = False

    def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        res = llm.predict("bar")
        assert res == "Upstage Bar"
    assert completed
