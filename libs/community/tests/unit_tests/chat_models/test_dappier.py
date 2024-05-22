"""Test EdenAI Chat API wrapper."""
from typing import List, cast

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import SecretStr

from langchain_community.chat_models.dappier import (
    ChatDappierAI,
    _format_dappier_messages,
)


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (
            [
                SystemMessage(
                    content="You are a chat model with real time search tools"
                ),
                HumanMessage(content="Hello how are you today?"),
            ],
            [
                {
                    "role": "system",
                    "content": "You are a chat model with real time search tools",
                },
                {"role": "user", "content": "Hello how are you today?"},
            ],
        )
    ],
)
def test_dappier_messages_formatting(
    messages: List[BaseMessage], expected: str
) -> None:
    result = _format_dappier_messages(messages)
    assert result == expected


def test_dappierai_initialization() -> None:
    for model in [
        ChatDappierAI(dappier_model="dappier-ai-model", dappier_api_key="xyz"),
        ChatDappierAI(model="dappier-ai-model", api_key="xyz"),
    ]:
        assert model.dappier_model == "dappier-ai-model"
        assert cast(SecretStr, model.dappier_api_key).get_secret_value() == "xyz"
