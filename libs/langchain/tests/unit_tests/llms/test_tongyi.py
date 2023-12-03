"""
This file is to perform unit tests for Tongyi llm.
"""

import pytest
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain.llms.tongyi import Tongyi

pytest.mark.requires("dashscope")


def test_api_key_is_secret_value() -> None:
    llm = Tongyi(dashscope_api_key="secret-key-api")
    assert isinstance(llm.dashscope_api_key, SecretStr)


@pytest.mark.requires("dashscope")
def test_api_key_is_secret_value_pass_by_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = Tongyi(dashscope_api_key="secret-key-api")
    print(llm.dashscope_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


@pytest.mark.requires("dashscope")
def test_api_key_is_secret_value_pass_by_end(
    capsys: CaptureFixture, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-api-key")
    llm = Tongyi()
    print(llm.dashscope_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
