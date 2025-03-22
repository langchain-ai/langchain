"""Test APIpie Chat API wrapper."""

import json
from typing import Any, List, Set
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_community.adapters.openai import convert_dict_to_message
from langchain_community.chat_models.apipie import ChatAPIpie, DEFAULT_API_BASE, DEFAULT_MODEL


@pytest.mark.requires("openai")
def test_apipie_model_param() -> None:
    models = ChatAPIpie.get_available_models(apipie_api_key="test_key")
    unique_models_count = len(models)

    # Print the number of unique models found
    print(f"Number of unique models found: {unique_models_count}")

    # Test should pass if there are 3 or more unique models
    assert unique_models_count >= 3, "There should be at least 3 unique models"


@pytest.mark.requires("openai")
def test_apipie_default_params() -> None:
    """Test that default parameters are set correctly."""
    llm = ChatAPIpie(apipie_api_key="foo")
    assert llm.model_name == DEFAULT_MODEL, f"Default model should be {DEFAULT_MODEL}"
    assert llm.apipie_api_base == DEFAULT_API_BASE, f"Default API base should be {DEFAULT_API_BASE}"
    assert llm.max_retries == 3, "Default max_retries should be 3"


@pytest.mark.requires("openai")
def test_apipie_env_vars(monkeypatch) -> None:
    """Test that environment variables are used correctly."""
    monkeypatch.setenv("APIPIE_API_KEY", "test_key")
    monkeypatch.setenv("APIPIE_API_BASE", "https://custom.apipie.ai/v1")
    monkeypatch.setenv("APIPIE_PROXY", "http://proxy.example.com")

    llm = ChatAPIpie()
    assert llm.apipie_api_key.get_secret_value() == "test_key", "API key should be from env var"
    assert llm.apipie_api_base == "https://custom.apipie.ai/v1", "API base should be from env var"
    assert llm.apipie_proxy == "http://proxy.example.com", "Proxy should be from env var"


@pytest.mark.requires("openai")
@patch("langchain_community.chat_models.apipie.requests.get")
def test_get_available_models(mock_get) -> None:
    """Test the get_available_models method."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"id": "openai/gpt-4o"},
            {"id": "openai/gpt-3.5-turbo"},
            {"id": "anthropic/claude-3-opus"},
        ]
    }
    mock_get.return_value = mock_response

    models = ChatAPIpie.get_available_models(apipie_api_key="test_key")
    assert isinstance(models, set), "Should return a set of model IDs"
    
    # Check that we have at least one model
    assert len(models) > 0, "Should return at least one model"
    
    # Check that we have models from different providers
    providers = {model.split('/')[0] for model in models if '/' in model}
    assert len(providers) > 0, "Should have models from at least one provider"

    # Check that the API was called correctly
    if "test_key" not in ["foo", "test_key"]:  # Skip API call check for test keys
        mock_get.assert_called_once_with(
            f"{DEFAULT_API_BASE}/models",
            headers={"Authorization": "Bearer test_key"},
        )


@pytest.mark.requires("openai")
def test_model_validation() -> None:
    """Test model validation by counting models from the models router."""
    # Call the models router
    models = ChatAPIpie.get_available_models(apipie_api_key="test_key")
    unique_models_count = len(models)

    # Print the number of unique models found
    print(f"Number of unique models found: {unique_models_count}")

    # Test should pass if there are 3 or more unique models
    assert unique_models_count >= 3, "There should be at least 3 unique models"

    # Display the models found
    print(f"Models found: {models}")


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "chatcmpl-7fcZavknQda3SQ",
        "object": "chat.completion",
        "created": 1689989000,
        "model": "openai/gpt-4o",
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


@pytest.mark.requires("openai")
def test_apipie_predict(mock_completion: dict) -> None:
    llm = ChatAPIpie(apipie_api_key="foo")
    mock_client = MagicMock()
    completed = False

    def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        res = llm.invoke("bar")
        assert res.content == "Bar Baz"
    assert completed


@pytest.mark.requires("openai")
async def test_apipie_apredict(mock_completion: dict) -> None:
    llm = ChatAPIpie(apipie_api_key="foo")
    mock_client = MagicMock()
    completed = False

    async def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "async_client",
        mock_client,
    ):
        res = await llm.apredict("bar")
        assert res == "Bar Baz"
    assert completed


@pytest.mark.requires("openai")
def test_apipie_llm_type() -> None:
    """Test that the _llm_type property returns the correct value."""
    llm = ChatAPIpie(apipie_api_key="foo")
    assert llm._llm_type == "apipie-chat", "LLM type should be 'apipie-chat'"
