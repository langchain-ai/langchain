import pytest
from pydantic import SecretStr


def test_create_chat_result_recovers_from_null_choices() -> None:
    """Test that _create_chat_result recovers when Pydantic sets choices=None
    but the raw dict has valid choices. Regression test for issue #32252."""
    from unittest.mock import MagicMock

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(api_key=SecretStr("fake-key"))

    # Simulate a Pydantic model where .choices is None but model_dump() has data
    mock_response = MagicMock()
    mock_response.choices = None
    mock_response.model_dump.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello from vLLM!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "model": "mistral-7b",
        "id": "chatcmpl-abc123",
    }
    mock_response.id = "chatcmpl-abc123"
    mock_response.model = "mistral-7b"

    result = llm._create_chat_result(mock_response)

    assert len(result.generations) == 1
    assert result.generations[0].message.content == "Hello from vLLM!"


def test_create_chat_result_raises_error_on_null_choices() -> None:
    """Test that a helpful ValueError is raised when choices is truly absent."""
    from unittest.mock import MagicMock

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(api_key=SecretStr("fake-key"))

    mock_response = MagicMock()
    mock_response.choices = None
    mock_response.model_dump.return_value = {"choices": None, "usage": None}

    with pytest.raises(ValueError, match="OpenAI-compatible"):
        llm._create_chat_result(mock_response)
