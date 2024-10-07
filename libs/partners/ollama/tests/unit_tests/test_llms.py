"""Test Ollama Chat API wrapper."""

from langchain_ollama import OllamaLLM


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
