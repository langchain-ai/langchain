"""Test Cleanlab's TLM"""

import pytest
from langchain_core.pydantic_v1 import SecretStr
from pytest import MonkeyPatch

from langchain_community.llms.cleanlab import TLM


def test_api_key_is_secret_string() -> None:
    tlm = TLM(cleanlab_api_key="secret-api-key")  # type: ignore[arg-type, call-arg]
    assert isinstance(tlm.cleanlab_api_key, SecretStr)
    assert tlm.cleanlab_api_key.get_secret_value() == "secret-api-key"


def test_api_key_masked_when_passed_via_constructor() -> None:
    tlm = TLM(cleanlab_api_key="secret-api-key")  # type: ignore[arg-type, call-arg]
    assert str(llm.cleanlab_api_key) == "**********"
    assert "secret-api-key" not in repr(llm.cleanlab_api_key)
    assert "secret-api-key" not in repr(llm)


def test_api_key_masked_when_passed_from_env() -> None:
    with MonkeyPatch.context() as mp:
        mp.setenv("CLEANLAB_API_KEY", "secret-api-key")
        tlm = TLM()  # type: ignore[call-arg]
        assert str(tlm.cleanlab_api_key) == "**********"
        assert "secret-api-key" not in repr(tlm.cleanlab_api_key)
        assert "secret-api-key" not in repr(tlm)
