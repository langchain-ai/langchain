"""Integration tests for ChatBocha chat model."""

import os

import pytest
from langchain_core.messages import HumanMessage

from langchain_bocha.chat_models import ChatBocha


@pytest.mark.skipif(
    not os.environ.get("BOCHA_API_KEY"),
    reason="BOCHA_API_KEY environment variable not set",
)
def test_chat_bocha_invoke() -> None:
    """Test ChatBocha invocation with a live query."""
    model = ChatBocha(model="deepseek-v4-pro")
    messages = [HumanMessage(content="Hello")]
    response = model.invoke(messages)
    assert response is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.skipif(
    not os.environ.get("BOCHA_API_KEY"),
    reason="BOCHA_API_KEY environment variable not set",
)
@pytest.mark.asyncio
async def test_chat_bocha_ainvoke() -> None:
    """Test ChatBocha asynchronous invocation with a live query."""
    model = ChatBocha(model="deepseek-v4-pro")
    messages = [HumanMessage(content="Hello")]
    response = await model.ainvoke(messages)
    assert response is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0
