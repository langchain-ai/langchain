"""Test Sarvam Chat API wrapper."""

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import langchain_core.load as lc_load
import pytest
from langchain_core.messages import AIMessage

from langchain_sarvam.chat_models import ChatSarvam

if "SARVAM_API_KEY" not in os.environ:
    os.environ["SARVAM_API_KEY"] = "fake-key"


def test_sarvam_model_param() -> None:
    llm = ChatSarvam(model="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"
    llm = ChatSarvam(model_name="bar")  # type: ignore[call-arg]
    assert llm.model_name == "bar"


def _mock_completion() -> dict:
    return {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "created": 1689989000,
        "model": "test-model",
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
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def test_sarvam_invoke() -> None:
    llm = ChatSarvam(model="foo")
    mock_client = MagicMock()
    completed = False

    def mock_completions(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return _mock_completion()

    mock_client.completions = mock_completions
    with patch.object(llm, "client", mock_client):
        res = llm.invoke("bar")
        assert res.content == "Bar Baz"
        assert type(res) is AIMessage
    assert completed


@pytest.mark.enable_socket
async def test_sarvam_ainvoke() -> None:
    llm = ChatSarvam(model="foo")
    mock_client = AsyncMock()
    completed = False

    async def mock_completions(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return _mock_completion()

    mock_client.completions = mock_completions
    with patch.object(llm, "async_client", mock_client):
        res = await llm.ainvoke("bar")
        assert res.content == "Bar Baz"
        assert type(res) is AIMessage
    assert completed


def test_chat_sarvam_invalid_streaming_params() -> None:
    with pytest.raises(ValueError):
        ChatSarvam(model="foo", streaming=True, n=2)


def test_chat_sarvam_secret() -> None:
    secret = "secretKey"  # noqa: S105
    not_secret = "safe"  # noqa: S105
    llm = ChatSarvam(model="foo", api_key=secret, model_kwargs={"not_secret": not_secret})  # type: ignore[call-arg, arg-type]
    stringified = str(llm)
    assert not_secret in stringified
    assert secret not in stringified


def test_sarvam_serialization() -> None:
    api_key1 = "top secret"
    api_key2 = "topest secret"
    llm = ChatSarvam(model="foo", api_key=api_key1, temperature=0.5)  # type: ignore[call-arg, arg-type]
    dump = lc_load.dumps(llm)
    llm2 = lc_load.loads(
        dump,
        valid_namespaces=["langchain_sarvam"],
        secrets_map={"SARVAM_API_KEY": api_key2},
    )

    assert type(llm2) is ChatSarvam

    assert llm.sarvam_api_key is not None
    assert llm.sarvam_api_key.get_secret_value() not in dump
    assert llm2.sarvam_api_key is not None
    assert llm2.sarvam_api_key.get_secret_value() == api_key2

    assert llm.temperature == llm2.temperature
