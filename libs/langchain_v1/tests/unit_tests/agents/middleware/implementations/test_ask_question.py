"""Unit tests for AskQuestionMiddleware."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from langchain.agents.factory import create_agent
from langchain.agents.middleware.ask_question import (
    ASK_QUESTION_SCHEMA_VERSION,
    ASK_QUESTION_SYSTEM_PROMPT,
    AskQuestion,
    AskQuestionInput,
    AskQuestionMiddleware,
    Question,
    QuestionOption,
    build_ask_question_interrupt_payload,
)
from langchain.agents.middleware.types import AgentState, ModelRequest, ModelResponse
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


def _fake_runtime() -> Runtime:
    return cast("Runtime", object())


def _sample_questions() -> list[Question]:
    return [
        Question(
            id="range",
            prompt="Which date range?",
            options=[
                QuestionOption(id="7d", label="Last 7 days"),
                QuestionOption(id="30d", label="Last 30 days"),
            ],
        )
    ]


def _make_request(system_prompt: str | None = None) -> ModelRequest:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
    return ModelRequest(
        model=model,
        system_prompt=system_prompt,
        messages=[],
        tool_choice=None,
        tools=[],
        response_format=None,
        state=AgentState(messages=[]),
        runtime=_fake_runtime(),
        model_settings={},
    )


def test_ask_question_middleware_initialization() -> None:
    middleware = AskQuestionMiddleware()
    assert len(middleware.tools) == 1
    assert middleware.tools[0].name == "AskQuestion"


def test_question_option_requires_free_text_placeholder() -> None:
    with pytest.raises(ValidationError):
        QuestionOption(id="custom", label="Custom", requires_free_text=True)


def test_question_rejects_duplicate_option_ids() -> None:
    with pytest.raises(ValidationError):
        Question(
            id="q1",
            prompt="Pick one",
            options=[
                QuestionOption(id="same", label="A"),
                QuestionOption(id="same", label="B"),
            ],
        )


def test_ask_question_input_rejects_duplicate_question_ids() -> None:
    questions = _sample_questions() + [
        Question(
            id="range",
            prompt="Duplicate id",
            options=[
                QuestionOption(id="a", label="A"),
                QuestionOption(id="b", label="B"),
            ],
        )
    ]
    with pytest.raises(ValidationError):
        AskQuestionInput(questions=questions)


def test_build_ask_question_interrupt_payload() -> None:
    questions = _sample_questions()
    payload = build_ask_question_interrupt_payload(
        questions=questions,
        title="Clarify request",
        context_note="Need your preference before continuing.",
    )
    assert payload["schema_version"] == ASK_QUESTION_SCHEMA_VERSION
    assert payload["title"] == "Clarify request"
    assert payload["context_note"] == "Need your preference before continuing."
    assert len(payload["questions"]) == 1
    assert payload["questions"][0]["id"] == "range"


def test_ask_question_tool_returns_resume_value() -> None:
    questions = _sample_questions()
    tool_call = {
        "args": {
            "questions": [question.model_dump() for question in questions],
            "title": "Clarify request",
        },
        "name": "AskQuestion",
        "type": "tool_call",
        "id": "test_call",
    }

    def mock_resume(_: dict[str, Any]) -> str:
        return '{"range": {"selected": ["7d"]}}'

    with patch(
        "langchain.agents.middleware.ask_question.interrupt",
        side_effect=mock_resume,
    ):
        result = AskQuestion.invoke(tool_call)

    assert result == '{"range": {"selected": ["7d"]}}'


def test_wrap_model_call_appends_system_prompt() -> None:
    middleware = AskQuestionMiddleware()
    request = _make_request(system_prompt="Base prompt")
    captured_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    middleware.wrap_model_call(request, mock_handler)
    assert captured_request is not None
    assert captured_request.system_prompt is not None
    assert "Base prompt" in captured_request.system_prompt
    assert ASK_QUESTION_SYSTEM_PROMPT.splitlines()[0] in captured_request.system_prompt


def test_parallel_ask_question_calls_rejected() -> None:
    middleware = AskQuestionMiddleware()
    ai_message = AIMessage(
        content="Need clarification",
        tool_calls=[
            {"name": "AskQuestion", "args": {"questions": []}, "id": "1"},
            {"name": "AskQuestion", "args": {"questions": []}, "id": "2"},
        ],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])
    result = middleware.after_model(state, _fake_runtime())
    assert result is not None
    assert len(result["messages"]) == 2
    assert all(isinstance(message, ToolMessage) for message in result["messages"])
    assert all(message.status == "error" for message in result["messages"])


def test_single_ask_question_call_allowed() -> None:
    middleware = AskQuestionMiddleware()
    ai_message = AIMessage(
        content="Need clarification",
        tool_calls=[{"name": "AskQuestion", "args": {"questions": []}, "id": "1"}],
    )
    state = AgentState[Any](messages=[HumanMessage(content="Hello"), ai_message])
    result = middleware.after_model(state, _fake_runtime())
    assert result is None


def test_agent_creation_with_ask_question_middleware() -> None:
    model = FakeToolCallingModel(tool_calls=[[], []])
    agent = create_agent(model=model, middleware=[AskQuestionMiddleware()])
    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert "messages" in result
