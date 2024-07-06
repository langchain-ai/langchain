"""Test Straico Chat API wrapper."""

import os

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.pydantic_v1 import SecretStr

from langchain_community.adapters.openai import (
    convert_dict_to_message,
)
from langchain_community.chat_models import ChatStraico

os.environ["STRAICO_API_KEY"] = "foo"


@pytest.mark.requires("openai")
def test_straico_model_name_param() -> None:
    llm = ChatStraico(model="foo")
    assert llm.model == "foo"


@pytest.mark.requires("openai")
def test_straico_initialization() -> None:
    """Test straico initialization."""
    # Verify that chat straico can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    for model in [
        ChatStraico(
            model="test",
            timeout=1,
            api_key=SecretStr("testkey"),
            verbose=True,
        ),
        ChatStraico(
            model="test",
            timeout=1,
            api_key=SecretStr("testkey"),
            verbose=True,
        ),
    ]:
        assert model.request_timeout == 1
        assert model.straico_api_key == SecretStr("testkey")


def test__convert_dict_to_message_human() -> None:
    message = {"role": "user", "content": "foo"}
    result = convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = convert_dict_to_message(message)
    expected_output = AIMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output
