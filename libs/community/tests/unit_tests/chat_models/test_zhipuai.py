from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_community.chat_models.zhipuai import ChatZhipuAI

zhipuai_api_key = "your_zhipu_api_key"


@pytest.mark.requires("zhipuai")
def test_chat_zhipuai_model_param() -> None:
    chat = ChatZhipuAI(
        zhipuai_api_key=zhipuai_api_key, model="chatglm_turbo", streaming=False
    )
    assert chat.model == "chatglm_turbo"
    assert chat.zhipuai_api_key == "your_zhipu_api_key"
    assert chat.streaming is False


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "zhipuai-test-123",
        "object": "chat.completion",
        "created": 1703222377,
        "model": "chatglm_turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Bar Baz",
                },
                "finish_reason": "stop",
            }
        ],
    }


@pytest.mark.requires("zhipuai")
def test_zhipuai_predict(mock_completion: dict) -> None:
    llm = ChatZhipuAI()
    mock_client = MagicMock()
    completed = False

    def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "_generate",
        mock_client,
    ):
        res = llm.predict("bar")
        assert res == "Bar Baz"
    assert completed


@pytest.mark.requires("zhipuai")
async def test_zhipuai_apredict(mock_completion: dict) -> None:
    llm = ChatZhipuAI()
    mock_client = MagicMock()
    completed = False

    def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "_agenerate",
        mock_client,
    ):
        res = llm.predict("bar")
        assert res == "Bar Baz"
    assert completed
