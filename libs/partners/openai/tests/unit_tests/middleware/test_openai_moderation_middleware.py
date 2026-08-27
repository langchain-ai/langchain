from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast
from unittest.mock import Mock

import pytest
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from openai.types.moderation import Moderation

from langchain_openai.middleware.openai_moderation import (
    OpenAIModerationError,
    OpenAIModerationMiddleware,
)

DEFAULT_OK_DATA: dict[str, Any] = {
    "flagged": False,
    "categories": {
        "harassment": False,
        "harassment/threatening": False,
        "hate": False,
        "hate/threatening": False,
        "illicit": False,
        "illicit/violent": False,
        "self-harm": False,
        "self-harm/instructions": False,
        "self-harm/intent": False,
        "sexual": False,
        "sexual/minors": False,
        "violence": False,
        "violence/graphic": False,
    },
    "category_scores": {
        "harassment": 0.0,
        "harassment/threatening": 0.0,
        "hate": 0.0,
        "hate/threatening": 0.0,
        "illicit": 0.0,
        "illicit/violent": 0.0,
        "self-harm": 0.0,
        "self-harm/instructions": 0.0,
        "self-harm/intent": 0.0,
        "sexual": 0.0,
        "sexual/minors": 0.0,
        "violence": 0.0,
        "violence/graphic": 0.0,
    },
    "category_applied_input_types": {
        "harassment": ["text"],
        "harassment/threatening": ["text"],
        "hate": ["text"],
        "hate/threatening": ["text"],
        "illicit": ["text"],
        "illicit/violent": ["text"],
        "self-harm": ["text"],
        "self-harm/instructions": ["text"],
        "self-harm/intent": ["text"],
        "sexual": ["text"],
        "sexual/minors": ["text"],
        "violence": ["text"],
        "violence/graphic": ["text"],
    },
}

DEFAULT_OK = Moderation.model_validate(DEFAULT_OK_DATA)


def flagged_result() -> Moderation:
    flagged_data = deepcopy(DEFAULT_OK_DATA)
    flagged_data["flagged"] = True
    flagged_data["categories"]["self-harm"] = True
    flagged_data["category_scores"]["self-harm"] = 0.9
    return Moderation.model_validate(flagged_data)


class StubModerationMiddleware(OpenAIModerationMiddleware):
    """Override OpenAI calls with deterministic fixtures."""

    def __init__(self, decisions: Mapping[str, Moderation], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._decisions = decisions

    def _moderate(self, text: str) -> Moderation:
        return self._decisions.get(text, DEFAULT_OK)

    async def _amoderate(self, text: str) -> Moderation:
        return self._moderate(text)


def make_state(
    messages: list[AIMessage | HumanMessage | ToolMessage],
) -> AgentState[Any]:
    return cast(AgentState[Any], {"messages": messages})


def test_before_model_allows_clean_input() -> None:
    middleware = StubModerationMiddleware({}, model="test")
    state = make_state([HumanMessage(content="hello")])

    assert middleware.before_model(state, Mock()) is None


def test_before_model_errors_on_flagged_input() -> None:
    middleware = StubModerationMiddleware(
        {"bad": flagged_result()}, model="test", exit_behavior="error"
    )
    state = make_state([HumanMessage(content="bad")])

    with pytest.raises(OpenAIModerationError) as exc:
        middleware.before_model(state, Mock())

    assert exc.value.result.flagged is True
    assert exc.value.stage == "input"


def test_before_model_jump_on_end_behavior() -> None:
    middleware = StubModerationMiddleware(
        {"bad": flagged_result()}, model="test", exit_behavior="end"
    )
    state = make_state([HumanMessage(content="bad")])

    response = middleware.before_model(state, Mock())

    assert response is not None
    assert response["jump_to"] == "end"
    ai_message = response["messages"][0]
    assert isinstance(ai_message, AIMessage)
    assert "flagged" in ai_message.content


def test_custom_violation_message_template() -> None:
    middleware = StubModerationMiddleware(
        {"bad": flagged_result()},
        model="test",
        exit_behavior="end",
        violation_message="Policy block: {categories}",
    )
    state = make_state([HumanMessage(content="bad")])

    response = middleware.before_model(state, Mock())

    assert response is not None
    assert response["messages"][0].content == "Policy block: self harm"


def test_after_model_replaces_flagged_message() -> None:
    middleware = StubModerationMiddleware(
        {"unsafe": flagged_result()}, model="test", exit_behavior="replace"
    )
    state = make_state([AIMessage(content="unsafe", id="ai-1")])

    response = middleware.after_model(state, Mock())
    assert response is not None
    updated_messages = response["messages"]
    assert isinstance(updated_messages[-1], AIMessage)
    assert updated_messages[-1].id == "ai-1"
    assert "flagged" in updated_messages[-1].content


def test_tool_messages_are_moderated_when_enabled() -> None:
    middleware = StubModerationMiddleware(
        {"dangerous": flagged_result()},
        model="test",
        check_tool_results=True,
        exit_behavior="replace",
    )
    state = make_state(
        [
            HumanMessage(content="question"),
            AIMessage(content="call tool"),
            ToolMessage(content="dangerous", tool_call_id="tool-1"),
        ]
    )

    response = middleware.before_model(state, Mock())
    assert response is not None
    updated_messages = response["messages"]
    tool_message = updated_messages[-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "tool-1"
    assert "flagged" in tool_message.content


@pytest.mark.asyncio
async def test_async_before_model_uses_async_moderation() -> None:
    middleware = StubModerationMiddleware(
        {"async": flagged_result()}, model="test", exit_behavior="end"
    )
    state = make_state([HumanMessage(content="async")])

    response = await middleware.abefore_model(state, Mock())
    assert response is not None
    assert response["jump_to"] == "end"
