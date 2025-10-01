"""Test ChatBaseten chat model."""

import os
import pytest
from langchain_core.messages import HumanMessage

from langchain_baseten import ChatBaseten


def test_chat_baseten_init() -> None:
    """Test ChatBaseten initialization."""
    chat = ChatBaseten(
        model="deepseek-ai/DeepSeek-V3-0324",
        baseten_api_key="test_key",
        temperature=0.7,
        max_tokens=100,
    )
    assert chat.model == "deepseek-ai/DeepSeek-V3-0324"
    assert chat.temperature == 0.7
    assert chat.max_tokens == 100


def test_chat_baseten_init_missing_api_key() -> None:
    """Test ChatBaseten initialization with missing API key."""
    # Ensure no API key is set in environment
    original_key = os.environ.get("BASETEN_API_KEY")
    if "BASETEN_API_KEY" in os.environ:
        del os.environ["BASETEN_API_KEY"]

    try:
        with pytest.raises(ValueError, match="You must specify an api key"):
            ChatBaseten(model="deepseek-ai/DeepSeek-V3-0324")
    finally:
        # Restore original key if it existed
        if original_key is not None:
            os.environ["BASETEN_API_KEY"] = original_key


def test_chat_baseten_params() -> None:
    """Test ChatBaseten parameters."""
    chat = ChatBaseten(
        model="deepseek-ai/DeepSeek-V3-0324",
        baseten_api_key="test_key",
        temperature=0.5,
        max_tokens=200,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.1,
    )

    params = chat._default_params
    assert params["model"] == "deepseek-ai/DeepSeek-V3-0324"
    assert params["temperature"] == 0.5
    assert params["max_tokens"] == 200
    assert params["top_p"] == 0.9
    assert params["frequency_penalty"] == 0.1
    assert params["presence_penalty"] == 0.1


def test_chat_baseten_identifying_params() -> None:
    """Test ChatBaseten identifying parameters."""
    chat = ChatBaseten(
        model="deepseek-ai/DeepSeek-V3-0324",
        baseten_api_key="test_key",
        temperature=0.5,
        max_tokens=200,
    )

    identifying_params = chat._identifying_params
    expected_params = {
        "model": "deepseek-ai/DeepSeek-V3-0324",
        "temperature": 0.5,
        "max_tokens": 200,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
    }
    assert identifying_params == expected_params


def test_chat_baseten_llm_type() -> None:
    """Test ChatBaseten LLM type."""
    chat = ChatBaseten(
        model="deepseek-ai/DeepSeek-V3-0324",
        baseten_api_key="test_key",
    )
    assert chat._llm_type == "baseten-chat"
