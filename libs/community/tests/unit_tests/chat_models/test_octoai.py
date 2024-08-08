import pytest
from pydantic import ValidationError
from langchain_core.pydantic_v1 import SecretStr
from langchain_community.chat_models.octoai import ChatOctoAI

DEFAULT_API_BASE = "https://text.octoai.run/v1/"
DEFAULT_MODEL = "llama-2-13b-chat"

def test__default_octoai_api_base() -> None:
    chat = ChatOctoAI(octoai_api_token=SecretStr("test_token"))
    assert chat.octoai_api_base == DEFAULT_API_BASE

def test__default_octoai_api_token() -> None:
    chat = ChatOctoAI(octoai_api_token=SecretStr("test_token"))
    assert chat.octoai_api_token.get_secret_value() == "test_token"

def test__default_model_name() -> None:
    chat = ChatOctoAI(octoai_api_token=SecretStr("test_token"))
    assert chat.model_name == DEFAULT_MODEL

def test__field_aliases() -> None:
    chat = ChatOctoAI(octoai_api_token=SecretStr("test_token"), model="custom-model")
    assert chat.model_name == "custom-model"
    assert chat.octoai_api_token.get_secret_value() == "test_token"

def test__missing_octoai_api_token() -> None:
    with pytest.raises(ValidationError) as e:
        ChatOctoAI()
    assert "Did not find octoai_api_token" in str(e)

def test__all_fields_provided() -> None:
    chat = ChatOctoAI(octoai_api_token=SecretStr("test_token"), model="custom-model", octoai_api_base="https://custom.api/base/")
    assert chat.octoai_api_base == "https://custom.api/base/"
    assert chat.octoai_api_token.get_secret_value() == "test_token"
    assert chat.model_name == "custom-model"
