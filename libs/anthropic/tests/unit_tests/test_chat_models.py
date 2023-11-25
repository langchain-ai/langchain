"""Test Anthropic Chat API wrapper."""
import os
from typing import List

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langchain_anthropic.chat_models import (
    ChatAnthropic,
    convert_messages_to_prompt_anthropic,
)

os.environ["ANTHROPIC_API_KEY"] = "foo"


def test_model_name_param() -> None:
    llm = ChatAnthropic(model_name="foo")
    assert llm.model == "foo"


def test_model_param() -> None:
    llm = ChatAnthropic(model="foo")
    assert llm.model == "foo"


def test_model_kwargs() -> None:
    llm = ChatAnthropic(model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


def test_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        ChatAnthropic(model_kwargs={"max_tokens_to_sample": 5})


def test_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = ChatAnthropic(foo="bar")
    assert llm.model_kwargs == {"foo": "bar"}


def test_initialization() -> None:
    """Test anthropic initialization."""
    # No params.
    ChatAnthropic()

    # All params.
    ChatAnthropic(
        model="test",
        max_tokens_to_sample=1000,
        temperature=0.2,
        top_k=2,
        top_p=0.9,
        default_request_timeout=123,
        anthropic_api_url="foobar.com",
        anthropic_api_key="test",
        model_kwargs={"fake_param": 2},
    )

    # Alias params
    ChatAnthropic(
        model_name="test",
        timeout=123,
        base_url="foobar.com",
        api_key="test",
    )


def test_get_num_tokens() -> None:
    chat = ChatAnthropic(model="test", anthropic_api_key="test")
    assert chat.get_num_tokens("Hello claude") > 0


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
