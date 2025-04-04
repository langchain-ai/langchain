"""Test Perplexity Chat API wrapper."""

import os
from typing import Any, Dict, List, Optional, Tuple, Type
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_tests.unit_tests import ChatModelUnitTests
from pytest_mock import MockerFixture

from langchain_community.chat_models import ChatPerplexity

os.environ["PPLX_API_KEY"] = "foo"


@pytest.mark.requires("openai")
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


@pytest.mark.requires("openai")
def test_perplexity_model_name_param() -> None:
    llm = ChatPerplexity(model="foo")  # type: ignore[call-arg]
    assert llm.model == "foo"


@pytest.mark.requires("openai")
def test_perplexity_model_kwargs() -> None:
    llm = ChatPerplexity(model="test", model_kwargs={"foo": "bar"})  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("openai")
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
        assert model.pplx_api_key == "test"


@pytest.mark.requires("openai")
def test_perplexity_stream_includes_citations(mocker: MockerFixture) -> None:
    """Test that the stream method includes citations in the additional_kwargs."""
    llm = ChatPerplexity(
        model="test",
        timeout=30,
        verbose=True,
    )
    mock_chunk_0 = {
        "choices": [
            {
                "delta": {
                    "content": "Hello ",
                },
                "finish_reason": None,
            }
        ],
        "citations": ["example.com", "example2.com"],
    }
    mock_chunk_1 = {
        "choices": [
            {
                "delta": {
                    "content": "Perplexity",
                },
                "finish_reason": None,
            }
        ],
        "citations": ["example.com", "example2.com"],
    }
    mock_chunks: List[Dict[str, Any]] = [mock_chunk_0, mock_chunk_1]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks
    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_stream
    )
    stream = llm.stream("Hello langchain")
    full: Optional[BaseMessageChunk] = None
    for i, chunk in enumerate(stream):
        full = chunk if full is None else full + chunk
        assert chunk.content == mock_chunks[i]["choices"][0]["delta"]["content"]
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

    patcher.assert_called_once()


@pytest.mark.requires("openai")
def test_perplexity_stream_includes_citations_and_images(mocker: MockerFixture) -> None:
    """Test that the stream method includes citations in the additional_kwargs."""
    llm = ChatPerplexity(
        model="test",
        timeout=30,
        verbose=True,
    )
    mock_chunk_0 = {
        "choices": [
            {
                "delta": {
                    "content": "Hello ",
                },
                "finish_reason": None,
            }
        ],
        "citations": ["example.com", "example2.com"],
        "images": [
            {
                "image_url": "mock_image_url",
                "origin_url": "mock_origin_url",
                "height": 100,
                "width": 100,
            }
        ],
    }
    mock_chunk_1 = {
        "choices": [
            {
                "delta": {
                    "content": "Perplexity",
                },
                "finish_reason": None,
            }
        ],
        "citations": ["example.com", "example2.com"],
        "images": [
            {
                "image_url": "mock_image_url",
                "origin_url": "mock_origin_url",
                "height": 100,
                "width": 100,
            }
        ],
    }
    mock_chunks: List[Dict[str, Any]] = [mock_chunk_0, mock_chunk_1]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks
    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_stream
    )
    stream = llm.stream("Hello langchain")
    full: Optional[BaseMessageChunk] = None
    for i, chunk in enumerate(stream):
        full = chunk if full is None else full + chunk
        assert chunk.content == mock_chunks[i]["choices"][0]["delta"]["content"]
        if i == 0:
            assert chunk.additional_kwargs["citations"] == [
                "example.com",
                "example2.com",
            ]
            assert chunk.additional_kwargs["images"] == [
                {
                    "image_url": "mock_image_url",
                    "origin_url": "mock_origin_url",
                    "height": 100,
                    "width": 100,
                }
            ]
        else:
            assert "citations" not in chunk.additional_kwargs
            assert "images" not in chunk.additional_kwargs
    assert isinstance(full, AIMessageChunk)
    assert full.content == "Hello Perplexity"
    assert full.additional_kwargs == {
        "citations": ["example.com", "example2.com"],
        "images": [
            {
                "image_url": "mock_image_url",
                "origin_url": "mock_origin_url",
                "height": 100,
                "width": 100,
            }
        ],
    }

    patcher.assert_called_once()


@pytest.mark.requires("openai")
def test_perplexity_stream_includes_citations_and_related_questions(
    mocker: MockerFixture,
) -> None:
    """Test that the stream method includes citations in the additional_kwargs."""
    llm = ChatPerplexity(
        model="test",
        timeout=30,
        verbose=True,
    )
    mock_chunk_0 = {
        "choices": [
            {
                "delta": {
                    "content": "Hello ",
                },
                "finish_reason": None,
            }
        ],
        "citations": ["example.com", "example2.com"],
        "related_questions": ["example_question_1", "example_question_2"],
    }
    mock_chunk_1 = {
        "choices": [
            {
                "delta": {
                    "content": "Perplexity",
                },
                "finish_reason": None,
            }
        ],
        "citations": ["example.com", "example2.com"],
        "related_questions": ["example_question_1", "example_question_2"],
    }
    mock_chunks: List[Dict[str, Any]] = [mock_chunk_0, mock_chunk_1]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks
    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_stream
    )
    stream = llm.stream("Hello langchain")
    full: Optional[BaseMessageChunk] = None
    for i, chunk in enumerate(stream):
        full = chunk if full is None else full + chunk
        assert chunk.content == mock_chunks[i]["choices"][0]["delta"]["content"]
        if i == 0:
            assert chunk.additional_kwargs["citations"] == [
                "example.com",
                "example2.com",
            ]
            assert chunk.additional_kwargs["related_questions"] == [
                "example_question_1",
                "example_question_2",
            ]
        else:
            assert "citations" not in chunk.additional_kwargs
            assert "related_questions" not in chunk.additional_kwargs
    assert isinstance(full, AIMessageChunk)
    assert full.content == "Hello Perplexity"
    assert full.additional_kwargs == {
        "citations": ["example.com", "example2.com"],
        "related_questions": ["example_question_1", "example_question_2"],
    }

    patcher.assert_called_once()
