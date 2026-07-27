"""Regression for token callbacks during agent ainvoke (issue #37878).

`create_agent` model execution used `invoke`/`ainvoke`, which never emits
`on_llm_new_token`. Planning still needs the full message, so we stream-then-join.
"""

from __future__ import annotations

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain.agents import create_agent


class TokenTracker(BaseCallbackHandler):
    def __init__(self) -> None:
        self.tokens: list[str] = []

    def on_llm_new_token(self, token: str, **kwargs: object) -> None:
        self.tokens.append(token)


@pytest.mark.asyncio
async def test_ainvoke_emits_on_llm_new_token() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello world")]))
    agent = create_agent(model, tools=[])
    tracker = TokenTracker()

    result = await agent.ainvoke(
        {"messages": [HumanMessage("hi")]},
        config={"callbacks": [tracker]},
    )

    assert tracker.tokens, "expected on_llm_new_token during agent.ainvoke"
    assert "".join(tracker.tokens) == "Hello world"
    assert result["messages"][-1].content == "Hello world"


def test_invoke_emits_on_llm_new_token() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello world")]))
    agent = create_agent(model, tools=[])
    tracker = TokenTracker()

    result = agent.invoke(
        {"messages": [HumanMessage("hi")]},
        config={"callbacks": [tracker]},
    )

    assert tracker.tokens, "expected on_llm_new_token during agent.invoke"
    assert "".join(tracker.tokens) == "Hello world"
    assert result["messages"][-1].content == "Hello world"
