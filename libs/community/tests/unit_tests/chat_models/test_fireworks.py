"""Test Fireworks chat model"""

import sys

import pytest
from pydantic import SecretStr
from pytest import CaptureFixture

from langchain_community.chat_models import ChatFireworks

if sys.version_info < (3, 9):
    pytest.skip("fireworks-ai requires Python > 3.8", allow_module_level=True)


@pytest.mark.requires("fireworks")
def test_api_key_is_string() -> None:
    llm = ChatFireworks(fireworks_api_key="secret-api-key")  # type: ignore[arg-type]
    assert isinstance(llm.fireworks_api_key, SecretStr)


@pytest.mark.requires("fireworks")
def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = ChatFireworks(fireworks_api_key="secret-api-key")  # type: ignore[arg-type]
    print(llm.fireworks_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"
