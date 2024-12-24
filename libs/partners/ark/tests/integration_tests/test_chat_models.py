"""Test ChatArk chat model."""

import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)

from langchain_ark import ChatArk

load_dotenv(override=True)


#
# Smoke test Runnable interface
#
@pytest.mark.scheduled
def test_invoke() -> None:
    """Test Chat wrapper."""
    chat = ChatArk(
        model=os.environ["ARK_CHAT_MODEL"],
        temperature=0.1,
    )  # type: ignore[call-arg]
    message = HumanMessage(content="Hello, Doubao")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)
