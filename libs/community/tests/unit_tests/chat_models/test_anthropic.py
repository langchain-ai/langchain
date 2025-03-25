"""Test Anthropic Chat API wrapper."""

import os
from typing import List

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langchain_community.chat_models import ChatAnthropic
from langchain_community.chat_models.anthropic import (
    convert_messages_to_prompt_anthropic,
)

os.environ["ANTHROPIC_API_KEY"] = "foo"


@pytest.mark.requires("anthropic")
def test_anthropic_model_name_param() -> None:
    llm = ChatAnthropic(model_name="foo")
    assert llm.model == "foo"


@pytest.mark.requires("anthropic")
def test_anthropic_model_param() -> None:
    llm = ChatAnthropic(model="foo")  # type: ignore[call-arg]
    assert llm.model == "foo"


@pytest.mark.requires("anthropic")
def test_anthropic_model_kwargs() -> None:
    llm = ChatAnthropic(model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("anthropic")
def test_anthropic_fields_in_model_kwargs() -> None:
    """Test that for backwards compatibility fields can be passed in as model_kwargs."""
    llm = ChatAnthropic(model_kwargs={"max_tokens_to_sample": 5})
    assert llm.max_tokens_to_sample == 5
    llm = ChatAnthropic(model_kwargs={"max_tokens": 5})
    assert llm.max_tokens_to_sample == 5


@pytest.mark.requires("anthropic")
def test_anthropic_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = ChatAnthropic(foo="bar")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("anthropic")
def test_anthropic_initialization() -> None:
    """Test anthropic initialization."""
    # Verify that chat anthropic can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    ChatAnthropic(model="test", anthropic_api_key="test")  # type: ignore[arg-type, call-arg]


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ([HumanMessage(content="Hello")], "\n\nHuman: Hello\n\nAssistant:"),
        (
            [HumanMessage(content="Hello"), AIMessage(content="Answer:")],
            "\n\nHuman: Hello\n\nAssistant: Answer:",
        ),
        (
            [
                SystemMessage(content="You're an assistant"),
                HumanMessage(content="Hello"),
                AIMessage(content="Answer:"),
            ],
            "You're an assistant\n\nHuman: Hello\n\nAssistant: Answer:",
        ),
    ],
)
def test_formatting(messages: List[BaseMessage], expected: str) -> None:
    result = convert_messages_to_prompt_anthropic(messages)
    assert result == expected
