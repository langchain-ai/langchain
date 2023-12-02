"""
This file is to perform unit tests for Tongyi chat model.
"""

import pytest
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain.chat_models.tongyi import ChatTongyi


@pytest.mark.requires("dashscope")
def test_api_key_is_secret_value() -> None:
    llm = ChatTongyi(dashscope_api_key="secret-api-key")
    assert isinstance(llm.dashscope_api_key, SecretStr)
