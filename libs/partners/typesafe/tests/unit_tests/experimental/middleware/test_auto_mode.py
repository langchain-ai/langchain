"""Tests for `AutoModeMiddleware`."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import httpx2
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import InputAgentState, omit_payload
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError
from typing_extensions import Self, override

import langchain_typesafe
from langchain_typesafe import NoulCriteria, experimental
from langchain_typesafe.client import TypeSafeInternalServerError
from langchain_typesafe.experimental.middleware import AutoModeMiddleware
from langchain_typesafe.experimental.middleware import __all__ as middleware_all
from langchain_typesafe.types import Noul

API_KEY = "test-api-key"
pytestmark = pytest.mark.asyncio


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
        """Return this model after accepting the agent's tools."""
        _ = (tools, tool_choice, kwargs)
        return self


def _model(
    *,
    tool_name: str = "delete_file",
    args: dict[str, Any] | None = None,
) -> _ToolCallingModel:
    return _ToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name=tool_name,
                            args=args or {"path": "/workspace/report.txt"},
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


def _response_payload(probability: float) -> dict[str, Any]:
    return {
        "model": "jev-latest",
        "answers": {"is_risky": {"type": "noul", "noul": probability}},
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


@asynccontextmanager
async def _middleware(
    probability: float,
    *,
    tools: Sequence[str | BaseTool],
    instructions: str | None = None,
    criteria: NoulCriteria | None = None,
    status_code: int = 200,
    observed_requests: list[dict[str, Any]] | None = None,
) -> AsyncIterator[AutoModeMiddleware]:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if observed_requests is not None:
            observed_requests.append(json.loads(request.content))
        if status_code != 200:
            return httpx2.Response(status_code, json={"error": "unavailable"})
        return httpx2.Response(200, json=_response_payload(probability))

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    async_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    kwargs: dict[str, Any] = {}
    if instructions is not None:
        kwargs["instructions"] = instructions
    if criteria is not None:
        kwargs["criteria"] = criteria
    with patch.dict(os.environ, {"TYPESAFE_API_KEY": API_KEY}):
        middleware = AutoModeMiddleware(
            tools=tools,
            **kwargs,
        )
    created_client = middleware.classifier.client
    created_async_client = middleware.classifier.async_client
    if created_client is not None:
        created_client.close()
    if created_async_client is not None:
        await created_async_client.aclose()
    middleware.classifier.client = client
    middleware.classifier.async_client = async_client
    try:
        yield middleware
    finally:
        client.close()
        await async_client.aclose()


async def _run_agent(
    middleware: AutoModeMiddleware,
    tool_instance: BaseTool,
    *,
    async_: bool,
    model: _ToolCallingModel | None = None,
    messages: list[Any] | None = None,
) -> dict[str, Any]:
    agent = create_agent(
        model or _model(),
        tools=[tool_instance],
        middleware=[middleware],
    )
    state = InputAgentState(
        messages=messages
        if messages is not None
        else [HumanMessage("Delete the temporary report.")]
    )
    if async_:
        return await agent.ainvoke(state)
    return agent.invoke(state)


def _tool_messages(result: dict[str, Any]) -> list[ToolMessage]:
    return [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]


async def test_middleware_constructs_configurable_risk_classifier() -> None:
    """Construct the internal Noul from caller-supplied criteria and instructions."""
    custom_criteria = NoulCriteria(
        true="The call modifies production data.",
        false="The call reads public data.",
    )
    async with _middleware(
        0.2,
        tools=["delete_file"],
        instructions="Assess production impact.",
        criteria=custom_criteria,
    ) as middleware:
        question = middleware.classifier.questions["is_risky"]

        assert question == Noul(
            instructions="Assess production impact.",
            criteria=custom_criteria,
        )


async def test_none_criteria_is_supported() -> None:
    """Allow callers to classify without outcome criteria."""
    async with _middleware(0.2, tools=["delete_file"]) as middleware:
        assert middleware.config.criteria is None
        assert middleware.classifier.questions["is_risky"].criteria is None


async def test_base_tool_name_is_inferred() -> None:
    """Accept BaseTool instances and infer their configured names."""
    tool_instance = _delete_tool([])

    async with _middleware(0.2, tools=[tool_instance]) as middleware:
        assert middleware._tool_names == {"delete_file"}


async def test_experimental_middleware_is_not_exported_from_root() -> None:
    """Experimental middleware requires the explicit middleware namespace."""
    assert "AutoModeMiddleware" not in langchain_typesafe.__all__
    assert not hasattr(experimental, "AutoModeMiddleware")


async def test_trace_policy_omits_classifier_context() -> None:
    """Middleware traces omit authorization context and tool arguments."""
    async with _middleware(0.2, tools=["delete_file"]) as middleware:
        assert middleware.trace_policy.process_inputs is omit_payload


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
    """Apply Auto Mode through complete synchronous and asynchronous agent runs."""
    executions: list[str] = []
    tool_instance = _delete_tool(executions)

    async with _middleware(
        probability,
        tools=[tool_instance],
    ) as middleware:
        result = await _run_agent(
            middleware,
            tool_instance,
            async_=async_,
        )

    [tool_message] = _tool_messages(result)
    assert tool_message.status == expected_status
    assert tool_message.tool_call_id == "call_123"
    assert executions == expected_executions


async def test_unlisted_tool_bypasses_classification() -> None:
    """Execute unlisted tools without sending a classifier request."""
    executions: list[str] = []
    tool_instance = _delete_tool(executions)
    observed_requests: list[dict[str, Any]] = []

    async with _middleware(
        0.9,
        tools=["another_tool"],
        observed_requests=observed_requests,
    ) as middleware:
        result = await _run_agent(middleware, tool_instance, async_=False)

    [tool_message] = _tool_messages(result)
    assert tool_message.status == "success"
    assert executions == ["/workspace/report.txt"]
    assert observed_requests == []


async def test_classifier_receives_user_context_and_raw_tool_call() -> None:
    """Send user authorization context and complete tool details to TypeSafe."""
    tool_instance = _delete_tool([])
    observed_requests: list[dict[str, Any]] = []

    async with _middleware(
        0.9,
        tools=[tool_instance],
        observed_requests=observed_requests,
    ) as middleware:
        await _run_agent(middleware, tool_instance, async_=False)

    [request] = observed_requests
    state = request["state"]
    assert state["messages"][0] == {
        "role": "user",
        "content": "Delete the temporary report.",
    }
    assert state["messages"][1]["role"] == "assistant"
    assert state["messages"][1]["tool_calls"][0]["function"]["name"] == "delete_file"
    assert state["tool_call"] == {
        "id": "call_123",
        "name": "delete_file",
        "args": {"path": "/workspace/report.txt"},
    }
    assert state["tool_description"] == "Delete a file at the supplied path."


async def test_classifier_context_is_limited_to_last_30_messages() -> None:
    """Bound conversation context while retaining assistant tool-call context."""
    tool_instance = _delete_tool([])
    observed_requests: list[dict[str, Any]] = []
    history = [HumanMessage(f"message {index}") for index in range(31)]

    async with _middleware(
        0.9,
        tools=[tool_instance],
        observed_requests=observed_requests,
    ) as middleware:
        await _run_agent(
            middleware,
            tool_instance,
            async_=False,
            messages=history,
        )

    messages = observed_requests[0]["state"]["messages"]
    assert len(messages) == 30
    assert messages[0] == {"role": "user", "content": "message 2"}
    assert messages[-1]["role"] == "assistant"


@pytest.mark.parametrize("async_", [False, True])
async def test_classifier_failure_terminates_agent_run(*, async_: bool) -> None:
    """Propagate classifier failures without executing the configured tool."""
    executions: list[str] = []
    tool_instance = _delete_tool(executions)

    async with _middleware(
        0.0,
        tools=[tool_instance],
        status_code=500,
    ) as middleware:
        with pytest.raises(TypeSafeInternalServerError, match="500"):
            await _run_agent(
                middleware,
                tool_instance,
                async_=async_,
            )

    assert executions == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tools": []},
        {"tools": "delete_file"},
    ],
)
async def test_invalid_configuration_is_rejected(kwargs: dict[str, Any]) -> None:
    """Validate tool configuration through Pydantic."""
    with pytest.raises(ValidationError):
        AutoModeMiddleware(**kwargs)


async def test_experimental_public_interface() -> None:
    """Expose Auto Mode alongside the model router middleware."""
    assert middleware_all == [
        "AutoModeMiddleware",
        "ModelChoice",
        "ModelRouterMiddleware",
    ]
