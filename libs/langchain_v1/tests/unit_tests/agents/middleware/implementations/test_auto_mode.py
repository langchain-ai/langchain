"""Tests for `AutoModeMiddleware`."""

import asyncio
from collections.abc import Sequence
from typing import Any, Self

import pytest
from langchain_core.decisions import (
    BaseDecisionModel,
    ClassifierResponse,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Question,
    State,
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import Field, ValidationError
from typing_extensions import override

from langchain.agents import create_agent
from langchain.agents.middleware import AutoModeMiddleware
from langchain.agents.middleware.auto_mode import _risk_questions
from langchain.agents.middleware.types import InputAgentState, omit_payload


class _RiskClassifier(BaseDecisionModel):
    """Decision model that records requests and returns a fixed probability."""

    model: str = "fake-model"
    probability: float = 0.0
    error: str | None = None
    requests: list[Any] = Field(default_factory=list)

    @property
    @override
    def _provider(self) -> str:
        return "fake"

    @override
    def _classify(self, state: State, questions: dict[str, Question]) -> ClassifierResponse:
        self.requests.append((state, questions))
        if self.error is not None:
            raise RuntimeError(self.error)
        return ClassifierResponse(
            model=self.model,
            answers={"is_risky": NoulAnswer(type="noul", noul=self.probability)},
        )


class _ToolCallingModel(GenericFakeChatModel):
    """Deterministic chat model that accepts tool binding."""

    @override
    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Self:
        return self


def _model() -> _ToolCallingModel:
    return _ToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="delete_file",
                            args={"path": "/workspace/report.txt"},
                            id="call_123",
                            type="tool_call",
                        )
                    ],
                ),
                AIMessage("done"),
            ]
        )
    )


def _delete_tool(executions: list[str]) -> BaseTool:
    @tool
    def delete_file(path: str) -> str:
        """Delete a file at the supplied path."""
        executions.append(path)
        return "deleted"

    return delete_file


async def _run_agent(
    middleware: AutoModeMiddleware,
    tool_instance: BaseTool,
    *,
    async_: bool = False,
    messages: list[Any] | None = None,
) -> dict[str, Any]:
    agent = create_agent(_model(), tools=[tool_instance], middleware=[middleware])
    state = InputAgentState(messages=messages or [HumanMessage("Delete the temporary report.")])
    if async_:
        return await agent.ainvoke(state)
    return await asyncio.to_thread(agent.invoke, state)


def _tool_messages(result: dict[str, Any]) -> list[ToolMessage]:
    return [message for message in result["messages"] if isinstance(message, ToolMessage)]


def test_risk_question_uses_configured_instructions_and_criteria() -> None:
    criteria = NoulCriteria(true="The call modifies data.", false="The call reads data.")
    middleware = AutoModeMiddleware(
        tools=["delete_file"],
        instructions="Assess the impact.",
        criteria=criteria,
        classifier=_RiskClassifier(),
    )

    assert _risk_questions(middleware.config)["is_risky"] == Noul(
        instructions="Assess the impact.", criteria=criteria
    )


def test_criteria_default_to_none() -> None:
    middleware = AutoModeMiddleware(tools=["delete_file"], classifier=_RiskClassifier())

    question = _risk_questions(middleware.config)["is_risky"]
    assert isinstance(question, Noul)
    assert question.criteria is None


def test_base_tool_name_is_inferred() -> None:
    middleware = AutoModeMiddleware(tools=[_delete_tool([])], classifier=_RiskClassifier())

    assert middleware._tool_names == {"delete_file"}


def test_trace_policy_omits_classifier_context() -> None:
    middleware = AutoModeMiddleware(tools=["delete_file"], classifier=_RiskClassifier())

    assert middleware.trace_policy.process_inputs is omit_payload


@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize(
    ("probability", "expected_status", "expected_executions"),
    [(0.2, "success", ["/workspace/report.txt"]), (0.5, "error", [])],
)
async def test_agent_executes_or_blocks_calls_by_probability(
    probability: float,
    expected_status: str,
    expected_executions: list[str],
    *,
    async_: bool,
) -> None:
    executions: list[str] = []
    tool_instance = _delete_tool(executions)
    middleware = AutoModeMiddleware(
        tools=[tool_instance], classifier=_RiskClassifier(probability=probability)
    )

    result = await _run_agent(middleware, tool_instance, async_=async_)

    [tool_message] = _tool_messages(result)
    assert tool_message.status == expected_status
    assert tool_message.tool_call_id == "call_123"
    assert executions == expected_executions


async def test_unlisted_tool_bypasses_classification() -> None:
    executions: list[str] = []
    tool_instance = _delete_tool(executions)
    classifier = _RiskClassifier(probability=0.9)
    middleware = AutoModeMiddleware(tools=["another_tool"], classifier=classifier)

    result = await _run_agent(middleware, tool_instance)

    [tool_message] = _tool_messages(result)
    assert tool_message.status == "success"
    assert executions == ["/workspace/report.txt"]
    assert classifier.requests == []


async def test_classifier_receives_messages_tool_call_and_description() -> None:
    tool_instance = _delete_tool([])
    classifier = _RiskClassifier(probability=0.9)
    middleware = AutoModeMiddleware(tools=[tool_instance], classifier=classifier)

    await _run_agent(middleware, tool_instance)

    [(state, _)] = classifier.requests
    assert state["messages"][0].text == "Delete the temporary report."
    assert state["messages"][1].tool_calls[0]["name"] == "delete_file"
    assert state["tool_call"] == {
        "id": "call_123",
        "name": "delete_file",
        "args": {"path": "/workspace/report.txt"},
    }
    assert state["tool_description"] == "Delete a file at the supplied path."


async def test_classifier_context_is_limited_to_last_30_messages() -> None:
    tool_instance = _delete_tool([])
    classifier = _RiskClassifier(probability=0.9)
    middleware = AutoModeMiddleware(tools=[tool_instance], classifier=classifier)

    await _run_agent(
        middleware,
        tool_instance,
        messages=[HumanMessage(f"message {index}") for index in range(31)],
    )

    messages = classifier.requests[0][0]["messages"]
    assert len(messages) == 30
    assert messages[0].text == "message 2"
    assert isinstance(messages[-1], AIMessage)


@pytest.mark.parametrize("async_", [False, True])
async def test_classifier_failure_terminates_agent_run(*, async_: bool) -> None:
    executions: list[str] = []
    tool_instance = _delete_tool(executions)
    middleware = AutoModeMiddleware(
        tools=[tool_instance], classifier=_RiskClassifier(error="unavailable")
    )

    with pytest.raises(RuntimeError, match="unavailable"):
        await _run_agent(middleware, tool_instance, async_=async_)

    assert executions == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tools": [], "classifier": _RiskClassifier()},
        {"tools": "delete_file", "classifier": _RiskClassifier()},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AutoModeMiddleware(**kwargs)
