"""Regression tests for `create_agent` input state typing."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState

from langchain.agents import create_agent


def test_invoke_accepts_messages_state_input() -> None:
    """`agent.invoke` accepts `MessagesState` with `AnyMessage` entries."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
    agent = create_agent(model=model)
    input_state: MessagesState = MessagesState(
        messages=[HumanMessage(content="hi")],
    )

    result = agent.invoke(input_state)
    assert "messages" in result
    assert len(result["messages"]) >= 1


def test_invoke_accepts_message_dict_input() -> None:
    """`agent.invoke` accepts message dict entries in input state."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
    agent = create_agent(model=model)

    result = agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
    assert "messages" in result
    assert len(result["messages"]) >= 1
