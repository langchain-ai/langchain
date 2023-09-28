"""Test Anthropic Chat API wrapper."""
import os
from typing import List

import pytest

from langchain.chat_models import ChatAnthropic
from langchain.chat_models.anthropic import convert_messages_to_prompt_anthropic
from langchain.schema import AIMessage, BaseMessage, HumanMessage

os.environ["ANTHROPIC_API_KEY"] = "foo"


@pytest.mark.requires("anthropic")
def test_anthropic_model_name_param() -> None:
    llm = ChatAnthropic(model_name="foo")
    assert llm.model == "foo"


@pytest.mark.requires("anthropic")
def test_anthropic_model_param() -> None:
    llm = ChatAnthropic(model="foo")
    assert llm.model == "foo"


@pytest.mark.requires("anthropic")
def test_anthropic_model_kwargs() -> None:
    llm = ChatAnthropic(model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("anthropic")
def test_anthropic_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        ChatAnthropic(model_kwargs={"max_tokens_to_sample": 5})


@pytest.mark.requires("anthropic")
def test_anthropic_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = ChatAnthropic(foo="bar")
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("anthropic")
def test_anthropic_initialization() -> None:
    """Test anthropic initialization."""
    # Verify that chat anthropic can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    ChatAnthropic(model="test", anthropic_api_key="test")


def test_formatting() -> None:
    messages: List[BaseMessage] = [HumanMessage(content="Hello")]
    result = convert_messages_to_prompt_anthropic(messages)
    assert result == "\n\nHuman: Hello\n\nAssistant:"

    messages = [HumanMessage(content="Hello"), AIMessage(content="Answer:")]
    result = convert_messages_to_prompt_anthropic(messages)
    assert result == "\n\nHuman: Hello\n\nAssistant: Answer:"
