from __future__ import annotations

from collections.abc import Mapping, Sequence
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

    def _moderate(self, text: str | Sequence[str]) -> Moderation | list[Moderation]:
        if isinstance(text, str):
            return self._decisions.get(text, DEFAULT_OK)
        return [self._decisions.get(t, DEFAULT_OK) for t in text]

    async def _amoderate(
        self, text: str | Sequence[str]
    ) -> Moderation | list[Moderation]:
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


def test_after_model_checks_tool_call_arguments() -> None:
    """Verify tool call arguments are moderated even if content is empty."""
    middleware = StubModerationMiddleware(
        {"dangerous_arg": flagged_result()}, model="test", exit_behavior="end"
    )
    ai_msg = AIMessage(
        id="ai-1",
        content="",
        tool_calls=[{"name": "test_tool", "args": {"input": "dangerous_arg"}, "id": "call-1"}],
    )
    state = make_state([HumanMessage(content="run tool"), ai_msg])

    response = middleware.after_model(state, Mock())
    assert response is not None
    assert response["jump_to"] == "end"
    replaced = response["messages"][0]
    assert isinstance(replaced, AIMessage)
    assert replaced.id == "ai-1"
    assert replaced.tool_calls == []
    assert "flagged" in replaced.content


def test_after_model_exit_end_replaces_message_in_place_without_tool_calls() -> None:
    """Verify exit_behavior='end' replaces the flagged message with same id and empty tool_calls."""
    middleware = StubModerationMiddleware(
        {"unsafe": flagged_result()}, model="test", exit_behavior="end"
    )
    ai_msg = AIMessage(
        id="ai-123",
        content="unsafe",
        tool_calls=[{"name": "tool_a", "args": {}, "id": "call-1"}],
    )
    state = make_state([HumanMessage(content="hello"), ai_msg])

    response = middleware.after_model(state, Mock())
    assert response is not None
    assert response["jump_to"] == "end"
    replaced = response["messages"][0]
    assert replaced.id == "ai-123"
    assert replaced.tool_calls == []


def test_after_model_exit_replace_strips_tool_calls() -> None:
    """Verify exit_behavior='replace' strips tool calls so they are not executed."""
    middleware = StubModerationMiddleware(
        {"unsafe": flagged_result()}, model="test", exit_behavior="replace"
    )
    ai_msg = AIMessage(
        id="ai-123",
        content="unsafe",
        tool_calls=[{"name": "tool_a", "args": {}, "id": "call-1"}],
    )
    state = make_state([HumanMessage(content="hello"), ai_msg])

    response = middleware.after_model(state, Mock())
    assert response is not None
    replaced = response["messages"][-1]
    assert replaced.id == "ai-123"
    assert replaced.tool_calls == []
