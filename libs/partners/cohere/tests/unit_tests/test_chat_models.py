"""Test chat model integration."""
import typing

import pytest

from langchain_cohere.chat_models import ChatCohere


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatCohere(cohere_api_key="test")


@pytest.mark.parametrize(
    "chat_cohere,expected",
    [
        pytest.param(ChatCohere(cohere_api_key="test"), {}, id="defaults"),
        pytest.param(
            ChatCohere(cohere_api_key="test", model="foo", temperature=1.0),
            {
                "model": "foo",
                "temperature": 1.0,
            },
            id="values are set",
        ),
    ],
)
def test_default_params(chat_cohere: ChatCohere, expected: typing.Dict) -> None:
    actual = chat_cohere._default_params
    assert expected == actual
