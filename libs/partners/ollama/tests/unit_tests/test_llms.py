"""Test Ollama Chat API wrapper."""

from typing import Any
from unittest.mock import patch

from langchain_ollama import OllamaLLM

MODEL_NAME = "llama3.1"


def test_initialization() -> None:
    """Test integration initialization."""
    OllamaLLM(model="llama3")


def test_model_params() -> None:
    # Test standard tracing params
    llm = OllamaLLM(model="llama3")
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "ollama",
        "ls_model_type": "llm",
        "ls_model_name": "llama3",
    }

    llm = OllamaLLM(model="llama3", num_predict=3)
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "ollama",
        "ls_model_type": "llm",
        "ls_model_name": "llama3",
        "ls_max_tokens": 3,
    }


@patch("langchain_ollama.llms.validate_model")
def test_validate_model_on_init(mock_validate_model: Any) -> None:
    """Test that the model is validated on initialization when requested."""
    # Test that validate_model is called when validate_model_on_init=True
    OllamaLLM(model=MODEL_NAME, validate_model_on_init=True)
    mock_validate_model.assert_called_once()
    mock_validate_model.reset_mock()

    # Test that validate_model is NOT called when validate_model_on_init=False
    OllamaLLM(model=MODEL_NAME, validate_model_on_init=False)
    mock_validate_model.assert_not_called()

    # Test that validate_model is NOT called by default
    OllamaLLM(model=MODEL_NAME)
    mock_validate_model.assert_not_called()
