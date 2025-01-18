"""Test Perplexity Chat API wrapper."""

import os
from unittest.mock import MagicMock

import pytest

from langchain_community.chat_models import ChatPerplexity

os.environ["PPLX_API_KEY"] = "foo"


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
def test_perplexity_stream_includes_citations(mocker) -> None:
    llm = ChatPerplexity()
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
    mock_chunks = [mock_chunk_0, mock_chunk_1]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks
    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_stream
    )
    stream = llm.stream("Hello langchain")
    for i, chunk in enumerate(stream):
        assert chunk.content == mock_chunks[i]["choices"][0]["delta"]["content"]
        if i == 0:
            assert chunk.additional_kwargs["citations"] == [
                "example.com",
                "example2.com",
            ]
        else:
            assert "citations" not in chunk.additional_kwargs

    patcher.assert_called_once()
