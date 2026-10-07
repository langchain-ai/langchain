"""Tests for `OpenAIAutoModeMiddleware`."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import InputAgentState, omit_payload
from langchain_core.decisions import (
    BaseDecisionModel,
    DecisionRequest,
    DecisionResponse,
    PredicateAnswer,
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import SecretStr
from typing_extensions import Self, override

from langchain_openai._compat import httpx
from langchain_openai.chat_models.base import OpenAIAPIError
from langchain_openai.decisions import OpenAIDecisions
from langchain_openai.middleware import OpenAIAutoModeMiddleware

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.LangChainBetaWarning"
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


def _predicate(probability: float) -> dict[str, Any]:
    return {"type": "predicate", "name": "is_risky", "probability": probability}


def _middleware(
    answer: dict[str, Any] | None = None,
    *,
    tools: Sequence[str | BaseTool] = ("delete_file",),
    status_code: int = 200,
    observed: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> OpenAIAutoModeMiddleware:
    def respond(request: httpx.Request) -> httpx.Response:
        if observed is not None:
            observed.append(json.loads(request.content))
        if status_code != 200:
            return httpx.Response(status_code, json={"error": {"message": "down"}})
        return httpx.Response(
            200,
            json={"model": "gpt-6-luna", "answers": [answer or _predicate(0.2)]},
        )

    async def arespond(request: httpx.Request) -> httpx.Response:
        return respond(request)

    decisions = OpenAIDecisions(
        model="gpt-6-luna",
        api_key=SecretStr("test-api-key"),
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(arespond)),
    )
    return OpenAIAutoModeMiddleware(tools=tools, model=decisions, **kwargs)


async def _run_agent(
    middleware: OpenAIAutoModeMiddleware,
    tool_instance: BaseTool,
    *,
    async_: bool,
    messages: list[Any] | None = None,
) -> dict[str, Any]:
    agent = create_agent(_model(), tools=[tool_instance], middleware=[middleware])
    state = InputAgentState(
        messages=messages or [HumanMessage("Delete the temporary report.")]
    )
    if async_:
        return await agent.ainvoke(state)
    return agent.invoke(state)


def _tool_message(result: dict[str, Any]) -> ToolMessage:
    [message] = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    return message


@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize(
    ("probability", "expected_status", "expected_executions"),
    [(0.2, "success", ["/workspace/report.txt"]), (0.9, "error", [])],
)
async def test_agent_executes_safe_calls_and_blocks_risky_calls(
    probability: float,
    expected_status: str,
    expected_executions: list[str],
    *,
    async_: bool,
) -> None:
    executions: list[str] = []
    tool_instance = _delete_tool(executions)
    middleware = _middleware(_predicate(probability), tools=[tool_instance])

    result = await _run_agent(middleware, tool_instance, async_=async_)

    tool_message = _tool_message(result)
    assert tool_message.status == expected_status
    assert tool_message.tool_call_id == "call_123"
    assert executions == expected_executions


@pytest.mark.parametrize("async_", [False, True])
async def test_refusal_blocks_the_call(*, async_: bool) -> None:
    executions: list[str] = []
    tool_instance = _delete_tool(executions)
    middleware = _middleware({"type": "refusal", "name": "is_risky"})

    result = await _run_agent(middleware, tool_instance, async_=async_)

    tool_message = _tool_message(result)
    assert tool_message.status == "error"
    assert "could not be assessed" in str(tool_message.content)
    assert executions == []


@pytest.mark.parametrize("async_", [False, True])
async def test_classification_failure_terminates_agent_run(*, async_: bool) -> None:
    executions: list[str] = []
    tool_instance = _delete_tool(executions)
    middleware = _middleware(status_code=500)

    with pytest.raises(OpenAIAPIError):
        await _run_agent(middleware, tool_instance, async_=async_)

    assert executions == []


async def test_unlisted_tool_bypasses_classification() -> None:
    executions: list[str] = []
    tool_instance = _delete_tool(executions)
    observed: list[dict[str, Any]] = []
    middleware = _middleware(_predicate(0.9), tools=["another_tool"], observed=observed)

    result = await _run_agent(middleware, tool_instance, async_=False)

    assert _tool_message(result).status == "success"
    assert executions == ["/workspace/report.txt"]
    assert observed == []


async def test_request_contains_user_context_and_tool_call() -> None:
    tool_instance = _delete_tool([])
    observed: list[dict[str, Any]] = []
    middleware = _middleware(
        tools=[tool_instance], observed=observed, instructions="Assess impact."
    )

    await _run_agent(middleware, tool_instance, async_=False)

    [request] = observed
    assert request["questions"] == [
        {"type": "predicate", "name": "is_risky", "instructions": "Assess impact."}
    ]
    state = json.loads(request["input"])
    assert state["messages"][0] == {
        "role": "user",
        "content": "Delete the temporary report.",
    }
    assert state["messages"][1]["tool_calls"][0]["function"]["name"] == "delete_file"
    assert state["tool_call"] == {
        "id": "call_123",
        "name": "delete_file",
        "args": {"path": "/workspace/report.txt"},
    }
    assert state["tool_description"] == "Delete a file at the supplied path."


async def test_context_is_limited_to_last_30_messages() -> None:
    tool_instance = _delete_tool([])
    observed: list[dict[str, Any]] = []
    history = [HumanMessage(f"message {index}") for index in range(31)]

    await _run_agent(
        _middleware(observed=observed), tool_instance, async_=False, messages=history
    )

    messages = json.loads(observed[0]["input"])["messages"]
    assert len(messages) == 30
    assert messages[0] == {"role": "user", "content": "message 2"}
    assert messages[-1]["role"] == "assistant"


async def test_media_is_replaced_with_placeholders() -> None:
    tool_instance = _delete_tool([])
    observed: list[dict[str, Any]] = []
    image_message = HumanMessage(
        content=[
            {"type": "text", "text": "Delete this report."},
            {"type": "image", "base64": "iVBORw0KGgo=", "mime_type": "image/png"},
        ]
    )

    await _run_agent(
        _middleware(observed=observed),
        tool_instance,
        async_=False,
        messages=[image_message],
    )

    assert "iVBORw0KGgo=" not in observed[0]["input"]
    content = json.loads(observed[0]["input"])["messages"][0]["content"]
    assert content == "Delete this report.\n[image omitted]"


def test_base_tool_name_is_inferred() -> None:
    middleware = _middleware(tools=[_delete_tool([])])

    assert middleware.tool_names == {"delete_file"}


def test_model_name_builds_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    middleware = OpenAIAutoModeMiddleware(tools=["delete_file"], model="gpt-6-luna")

    assert middleware.decisions.model == "gpt-6-luna"


def test_is_beta() -> None:
    assert (OpenAIAutoModeMiddleware.__doc__ or "").startswith(".. beta::")


def test_trace_policy_omits_classification_context() -> None:
    assert _middleware().trace_policy.process_inputs is omit_payload


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tools": []},
        {"tools": "delete_file"},
        {"tools": ["delete_file"], "instructions": "  "},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must"):
        OpenAIAutoModeMiddleware(model="gpt-6-luna", **kwargs)


class _FixedRiskModel(BaseDecisionModel):
    """Non-OpenAI decision model returning a fixed risk probability."""

    probability: float

    @property
    @override
    def _provider(self) -> str:
        return "fake"

    @override
    def _decide(self, request: DecisionRequest) -> DecisionResponse:
        return DecisionResponse(
            model=self.model,
            answers={
                "is_risky": PredicateAnswer(
                    type="predicate", probability=self.probability
                )
            },
        )


@pytest.mark.parametrize("async_", [False, True])
async def test_any_decision_model_can_classify(*, async_: bool) -> None:
    executions: list[str] = []
    tool_instance = _delete_tool(executions)
    middleware = OpenAIAutoModeMiddleware(
        tools=[tool_instance],
        model=_FixedRiskModel(model="fake", probability=0.9),
    )

    result = await _run_agent(middleware, tool_instance, async_=async_)

    assert _tool_message(result).status == "error"
    assert executions == []
