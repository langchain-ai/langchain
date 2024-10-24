"""Test Cleanlab's TrustworthyLanguageModel"""

from pydantic import SecretStr
from pytest import MonkeyPatch

from langchain_community.llms.cleanlab import TrustworthyLanguageModel


def test_api_key_is_secret_string() -> None:
    tlm = TrustworthyLanguageModel(cleanlab_api_key="secret-api-key")  # type: ignore[arg-type, call-arg]
    assert isinstance(tlm.cleanlab_api_key, SecretStr)
    assert tlm.cleanlab_api_key.get_secret_value() == "secret-api-key"


def test_api_key_masked_when_passed_via_constructor() -> None:
    tlm = TrustworthyLanguageModel(cleanlab_api_key="secret-api-key")  # type: ignore[arg-type, call-arg]
    assert str(tlm.cleanlab_api_key) == "**********"
    assert "secret-api-key" not in repr(tlm.cleanlab_api_key)
    assert "secret-api-key" not in repr(tlm)


def test_api_key_masked_when_passed_from_env() -> None:
    with MonkeyPatch.context() as mp:
        mp.setenv("CLEANLAB_API_KEY", "secret-api-key")
        tlm = TrustworthyLanguageModel()  # type: ignore[call-arg]
        assert str(tlm.cleanlab_api_key) == "**********"
        assert "secret-api-key" not in repr(tlm.cleanlab_api_key)
        assert "secret-api-key" not in repr(tlm)
