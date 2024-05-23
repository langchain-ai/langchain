"""Test Everly AI Chat API wrapper."""
import os

import pytest
from langchain_core.pydantic_v1 import SecretStr, ValidationError

from langchain_community.chat_models import ChatEverlyAI

os.environ["EVERLYAI_API_KEY"] = "foo"
DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"


@pytest.mark.requires("openai")
def test_everlyai_chat_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify validation error if no api key found"""
    monkeypatch.delenv("EVERLYAI_API_KEY", raising=False)
    with pytest.raises(ValidationError) as e:
        ChatEverlyAI()
    assert "Did not find everlyai_api_key" in str(e)


@pytest.mark.requires("openai")
def test_everlyai_chat_default_params() -> None:
    """Check default parameters with environment API key"""
    chat = ChatEverlyAI()
    assert chat.everlyai_api_key is None
    assert chat.model_name == DEFAULT_MODEL
    assert chat.everlyai_api_base == "https://everlyai.xyz/hosted"
    assert chat.available_models == {
        "meta-llama/Llama-2-13b-chat-hf-quantized",
        "meta-llama/Llama-2-7b-chat-hf",
    }


@pytest.mark.requires("openai")
def test_everlyai_chat_param_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check use of parameter API key instead of environment API key"""
    monkeypatch.delenv("EVERLYAI_API_KEY", raising=False)
    chat = ChatEverlyAI(everlyai_api_key="test")
    assert isinstance(chat.everlyai_api_key, SecretStr)


@pytest.mark.requires("openai")
def test_everlyai_chat_initialization() -> None:
    """Ensure parameter names can be referenced by alias"""
    for model in [
        ChatEverlyAI(
            everlyai_api_key="test",
            model_name=DEFAULT_MODEL,
        ),
        ChatEverlyAI(
            api_key="test",
            model=DEFAULT_MODEL,
        ),
    ]:
        assert model.everlyai_api_key.get_secret_value() == "test"
        assert model.model_name == DEFAULT_MODEL
