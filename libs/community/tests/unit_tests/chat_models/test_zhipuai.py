from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_community.chat_models.zhipuai import ChatZhipuAI


def test_chat_zhipuai_model_param() -> None:
    chat = ChatZhipuAI(api_key="your_api_key", model="chatglm_turbo", streaming=False)
    assert chat.model == "chatglm_turbo"
    assert chat.api_key == "your_api_key"
    assert chat.streaming is False


@pytest.fixture
def mock_chat() -> dict:
    return {
        "code": 200,
        "msg": "",
        "success": True,
        "data": {
            "task_id": "75931252186628016897601864755556524089",
            "request_id": "123445676789",
            "task_status": "SUCCESS",
            "choices": [{"role": "assistant", "content": "我爱编程"}],
            "usage": {
                "prompt_tokens": 215,
                "completion_tokens": 302,
                "total_tokens": 517,
            },
        },
    }


@pytest.mark.requires("zhipuai")
def test_zhipuai_chat(mock_chat: dict) -> None:
    chat = ChatZhipuAI(api_key="your_api_key", model="chatglm_turbo", streaming=False)
    mock_client = MagicMock()
    completed = False

    def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_chat

    mock_client.create = mock_create
    with patch.object(
        chat,
        "client",
        mock_client,
    ):
        messages = [
            AIMessage(content="Hi."),
            SystemMessage(content="You are a helpful assistant as a translator."),
            HumanMessage(
                content="""Translate this sentence from English to Chinese:
    I love programming."""
            ),
        ]
        res = chat(messages)
        assert res == "我爱编程"
    assert completed
