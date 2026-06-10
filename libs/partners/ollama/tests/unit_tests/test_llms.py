"""Test Ollama Chat API wrapper."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_ollama import OllamaLLM

MODEL_NAME = "llama3.1"


def test_initialization() -> None:
    """Test integration initialization."""
    OllamaLLM(model=MODEL_NAME)


def test_model_params() -> None:
    """Test standard tracing params"""
    llm = OllamaLLM(model=MODEL_NAME)
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "ollama",
        "ls_model_type": "llm",
        "ls_model_name": MODEL_NAME,
    }

    llm = OllamaLLM(model=MODEL_NAME, num_predict=3)
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "ollama",
        "ls_model_type": "llm",
        "ls_model_name": MODEL_NAME,
        "ls_max_tokens": 3,
    }


@patch("langchain_ollama.llms.validate_model")
def test_validate_model_on_init(mock_validate_model: Any) -> None:
    """Test that the model is validated on initialization when requested."""
    OllamaLLM(model=MODEL_NAME, validate_model_on_init=True)
    mock_validate_model.assert_called_once()
    mock_validate_model.reset_mock()

    OllamaLLM(model=MODEL_NAME, validate_model_on_init=False)
    mock_validate_model.assert_not_called()
    OllamaLLM(model=MODEL_NAME)
    mock_validate_model.assert_not_called()


def test_reasoning_aggregation() -> None:
    """Test that reasoning chunks are aggregated into final response."""
    llm = OllamaLLM(model=MODEL_NAME, reasoning=True)
    prompts = ["some prompt"]
    mock_stream = [
        {"thinking": "I am thinking.", "done": False},
        {"thinking": " Still thinking.", "done": False},
        {"response": "Final Answer.", "done": True},
    ]

    with patch.object(llm, "_create_generate_stream") as mock_stream_method:
        mock_stream_method.return_value = iter(mock_stream)
        result = llm.generate(prompts)

    assert result.generations[0][0].generation_info is not None
    assert (
        result.generations[0][0].generation_info["thinking"]
        == "I am thinking. Still thinking."
    )


def test_create_generate_stream_raises_when_client_none() -> None:
    """Test that _create_generate_stream raises RuntimeError when client is None."""
    with patch("langchain_ollama.llms.Client") as mock_client_class:
        mock_client_class.return_value = MagicMock()
        llm = OllamaLLM(model="test-model")
        llm._client = None  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="sync client is not initialized"):
            list(llm._create_generate_stream("Hello"))


async def test_acreate_generate_stream_raises_when_client_none() -> None:
    """Test that _acreate_generate_stream raises RuntimeError when client is None."""
    with patch("langchain_ollama.llms.AsyncClient") as mock_client_class:
        mock_client_class.return_value = MagicMock()
        llm = OllamaLLM(model="test-model")
        llm._async_client = None  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="async client is not initialized"):
            async for _ in llm._acreate_generate_stream("Hello"):
                pass
