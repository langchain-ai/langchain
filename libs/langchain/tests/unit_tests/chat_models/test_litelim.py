"""Test litellm functionality"""
import pytest
from langchain_core.pydantic_v1 import SecretStr
from pytest import MonkeyPatch

from langchain.chat_models import litellm


@pytest.mark.requires("litellm")
def test_api_key_is_secret() -> None:
    pytest.importorskip("google.generativeai")
    llm = litellm.ChatLiteLLM(model_name="gpt2")
    assert isinstance(llm.model_name, SecretStr)
    assert llm.model_name.get_secret_value() == "gpt2"


@pytest.mark.requires("litellm")
def test_api_key_masked_when_passed_via_constructor() -> None:
    pytest.importorskip("google.generativeai")
    llm = litellm.ChatLiteLLM(model_name="gpt2")

    assert str(llm.model_name) == "**********"
    assert "gpt2" not in repr(llm.model_name)
    assert "gpt2" not in repr(llm)
    assert "gpt2" not in str(llm)


@pytest.mark.requires("litellm")
def test_api_key_masked_when_passed_via_env_var(monkeypatch: MonkeyPatch) -> None:
    pytest.importorskip("google.generativeai")
    monkeypatch.setenv("LITELLM_API_KEY", "gpt2")
    llm = litellm.ChatLiteLLM(
        params={"temperature": 0.1},
    )

    assert str(llm.model_name) == "**********"
    assert "gpt2" not in repr(llm.model_name)
    assert "gpt2" not in repr(llm)
    assert "gpt2" not in str(llm)
