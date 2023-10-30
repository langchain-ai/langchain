import pytest
from pytest import MonkeyPatch

from langchain.chat_models.konko import ChatKonko
from langchain.pydantic_v1 import SecretStr


@pytest.mark.requires("konko")
def test_api_key_is_secret_string_and_matches_input() -> None:
    llm = ChatKonko(
        openai_api_key="secret-openai-api-key", konko_api_key="secret-konko-api-key"
    )
    assert isinstance(llm.openai_api_key, SecretStr)
    assert isinstance(llm.konko_api_key, SecretStr)
    assert llm.openai_api_key.get_secret_value() == "secret-openai-api-key"
    assert llm.konko_api_key.get_secret_value() == "secret-konko-api-key"


@pytest.mark.requires("konko")
def test_api_key_masked_when_passed_via_constructor() -> None:
    llm = ChatKonko(
        openai_api_key="secret-openai-api-key", konko_api_key="secret-konko-api-key"
    )
    assert str(llm.openai_api_key) == "**********"
    assert str(llm.konko_api_key) == "**********"
    assert "secret-openai-api-key" not in repr(llm.openai_api_key)
    assert "secret-konko-api-key" not in repr(llm.konko_api_key)
    assert "secret-openai-api-key" not in repr(llm)
    assert "secret-konko-api-key" not in repr(llm)


@pytest.mark.requires("konko")
def test_api_key_masked_when_passed_via_env() -> None:
    with MonkeyPatch.context() as mp:
        mp.setenv("KONKO_API_KEY", "secret-konko-api-key")
        llm = ChatKonko()
        assert str(llm.konko_api_key) == "**********"
        assert "secret-konko-api-key" not in repr(llm.konko_api_key)
        assert "secret-konko-api-key" not in repr(llm)
