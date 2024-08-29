"""Test chat model integration."""
import json
import os
from typing import Any, AsyncGenerator, Generator, cast
from unittest.mock import patch

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.pydantic_v1 import SecretStr

from langchain_community.chat_models import ChatClovaX
from langchain_community.chat_models.naver import (
    _convert_message_to_naver_chat_message,
    _convert_naver_chat_message_to_message,
)

os.environ["NCP_CLOVASTUDIO_API_KEY"] = "test_api_key"
os.environ["NCP_APIGW_API_KEY"] = "test_gw_key"


def test_initialization_api_key() -> None:
    """Test chat model initialization."""
    chat_model = ChatClovaX(api_key="foo", apigw_api_key="bar")
    assert (
        cast(SecretStr, chat_model.ncp_clovastudio_api_key).get_secret_value() == "foo"
    )
    assert cast(SecretStr, chat_model.ncp_apigw_api_key).get_secret_value() == "bar"


def test_initialization_model_name() -> None:
    llm = ChatClovaX(model="HCX-DASH-001")
    assert llm.model_name == "HCX-DASH-001"
    llm = ChatClovaX(model_name="HCX-DASH-001")
    assert llm.model_name == "HCX-DASH-001"


def test_convert_dict_to_message_human() -> None:
    message = {"role": "user", "content": "foo"}
    result = _convert_naver_chat_message_to_message(message)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_naver_chat_message(expected_output) == message


def test_convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = _convert_naver_chat_message_to_message(message)
    expected_output = AIMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_naver_chat_message(expected_output) == message


def test_convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = _convert_naver_chat_message_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_naver_chat_message(expected_output) == message


@pytest.fixture
def mock_chat_completion_response() -> dict:
    return {
        "status": {"code": "20000", "message": "OK"},
        "result": {
            "message": {
                "role": "assistant",
                "content": "Phrases: Record what happened today and prepare "
                "for tomorrow. "
                "The diary will make your life richer.",
            },
            "stopReason": "LENGTH",
            "inputLength": 100,
            "outputLength": 10,
            "aiFilter": [
                {"groupName": "curse", "name": "insult", "score": "1"},
                {"groupName": "curse", "name": "discrimination", "score": "0"},
                {
                    "groupName": "unsafeContents",
                    "name": "sexualHarassment",
                    "score": "2",
                },
            ],
        },
    }


def test_naver_invoke(mock_chat_completion_response: dict) -> None:
    llm = ChatClovaX()
    completed = False

    def mock_completion_with_retry(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_chat_completion_response

    with patch.object(ChatClovaX, "_completion_with_retry", mock_completion_with_retry):
        res = llm.invoke("Let's test it.")
        assert (
            res.content
            == "Phrases: Record what happened today and prepare for tomorrow. "
            "The diary will make your life richer."
        )
    assert completed


async def test_naver_ainvoke(mock_chat_completion_response: dict) -> None:
    llm = ChatClovaX()
    completed = False

    async def mock_acompletion_with_retry(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_chat_completion_response

    with patch.object(
        ChatClovaX, "_acompletion_with_retry", mock_acompletion_with_retry
    ):
        res = await llm.ainvoke("Let's test it.")
        assert (
            res.content
            == "Phrases: Record what happened today and prepare for tomorrow. "
            "The diary will make your life richer."
        )
    assert completed


def _make_completion_response_from_token(token: str):
    from httpx_sse import ServerSentEvent
    return ServerSentEvent(
        event="token",
        data=json.dumps(
            dict(
                index=0,
                inputLength=89,
                outputLength=1,
                message=dict(
                    content=token,
                    role="assistant",
                ),
            )
        ),
    )


def mock_chat_stream(*args: Any, **kwargs: Any) -> Generator:
    def it() -> Generator:
        for token in ["Hello", " how", " can", " I", " help", "?"]:
            yield _make_completion_response_from_token(token)

    return it()


async def mock_chat_astream(*args: Any, **kwargs: Any) -> AsyncGenerator:
    async def it() -> AsyncGenerator:
        for token in ["Hello", " how", " can", " I", " help", "?"]:
            yield _make_completion_response_from_token(token)

    return it()


class MyCustomHandler(BaseCallbackHandler):
    last_token: str = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.last_token = token


@patch(
    "langchain_community.chat_models.ChatClovaX._completion_with_retry",
    new=mock_chat_stream,
)
@pytest.mark.requires("httpx_sse")
def test_stream_with_callback() -> None:
    callback = MyCustomHandler()
    chat = ChatClovaX(callbacks=[callback])
    for token in chat.stream("Hello"):
        assert callback.last_token == token.content


@patch(
    "langchain_community.chat_models.ChatClovaX._acompletion_with_retry",
    new=mock_chat_astream,
)
@pytest.mark.requires("httpx_sse")
async def test_astream_with_callback() -> None:
    callback = MyCustomHandler()
    chat = ChatClovaX(callbacks=[callback])
    async for token in chat.astream("Hello"):
        assert callback.last_token == token.content
