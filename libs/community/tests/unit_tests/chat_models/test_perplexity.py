"""Test Perplexity Chat API wrapper."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Type
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessageChunk,
    HumanMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import SecretStr
from pytest_mock import MockerFixture
from requests.models import Response

from langchain_community.chat_models import ChatPerplexity

os.environ["PPLX_API_KEY"] = "foo"


class TestPerplexityStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatPerplexity

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        return (
            {"PPLX_API_KEY": "api_key"},
            {},
            {"pplx_api_key": "api_key"},
        )


def test_perplexity_model_name_param() -> None:
    llm = ChatPerplexity(model="foo")  # type: ignore[call-arg]
    assert llm.model == "foo"


def test_perplexity_model_kwargs() -> None:
    llm = ChatPerplexity(model="test", model_kwargs={"foo": "bar"})  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


def test_perplexity_initialization() -> None:
    """Test perplexity initialization."""
    # Verify that chat perplexity can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    for model in [
        ChatPerplexity(  # type: ignore[call-arg]
            model="test", timeout=1, api_key="test", temperature=0.7, verbose=True
        ),
        ChatPerplexity(  # type: ignore[call-arg]
            model="test",
            request_timeout=1,
            pplx_api_key="test",
            temperature=0.7,
            verbose=True,
        ),
    ]:
        assert model.request_timeout == 1
        assert isinstance(model.pplx_api_key, SecretStr)
        assert model.temperature == 0.7
        assert model.verbose is True


def test_perplexity_stream_includes_citations(mocker) -> None:
    """Test that the stream method includes citations in the additional_kwargs."""
    llm = ChatPerplexity(
        model="test",
        timeout=30,
        verbose=True,
    )

    mock_chunk_0 = {
        "choices": [
            {
                "delta": {"content": "Hello "},
                "finish_reason": None,
            }
        ],
        "citations": ["example.com", "example2.com"],
    }
    mock_chunk_1 = {
        "choices": [
            {
                "delta": {"content": "Perplexity"},
                "finish_reason": None,
            }
        ],
        "citations": ["example.com", "example2.com"],
    }

    mock_response = MagicMock(spec=Response)
    mock_response.iter_lines.return_value = [
        f"data: {json.dumps(mock_chunk_0)}".encode("utf-8"),
        f"data: {json.dumps(mock_chunk_1)}".encode("utf-8"),
    ]

    mock_post = mocker.patch("requests.post", return_value=mock_response)
    stream = llm.stream("Hello langchain")

    full: Optional[BaseMessageChunk] = None
    for i, chunk in enumerate(stream):
        full = chunk if full is None else full + chunk
        assert (
            chunk.content
            == [mock_chunk_0, mock_chunk_1][i]["choices"][0]["delta"]["content"]
        )

        if i == 0:
            assert chunk.additional_kwargs["citations"] == [
                "example.com",
                "example2.com",
            ]
        else:
            assert "citations" not in chunk.additional_kwargs

    assert isinstance(full, AIMessageChunk)
    assert full.content == "Hello Perplexity"
    assert full.additional_kwargs == {"citations": ["example.com", "example2.com"]}

    mock_post.assert_called_once()


def test_perplexity_generate(mocker) -> None:
    """Test the generate method."""
    llm = ChatPerplexity(
        model="test",
        timeout=30,
        verbose=True,
    )

    mock_response_data = {
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 226,
            "total_tokens": 234,
            "citation_tokens": 123,
            "num_search_queries": 1,
        },
        "citations": ["example.com", "example2.com"],
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Hello Perplexity",
                },
                "delta": {"role": "assistant", "content": ""},
            }
        ],
    }

    mock_response = MagicMock(spec=Response)
    mock_response.json.return_value = mock_response_data

    mocker.patch("requests.post", return_value=mock_response)

    msg = HumanMessage(content="Hi")
    result = llm._generate([msg])

    assert isinstance(result, ChatResult)
    assert len(result.generations) == 1
    assert isinstance(result.generations[0], ChatGeneration)
    assert isinstance(result.generations[0].message, AIMessage)
    assert result.generations[0].message.content.startswith("Hello Perplexity")
    assert "citations" in result.generations[0].message.additional_kwargs
    assert len(result.generations[0].message.additional_kwargs["citations"]) == 2
