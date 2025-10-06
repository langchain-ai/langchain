"""Unit tests for on_tool_call middleware hook."""

from collections.abc import Generator
from typing import Any, Literal, Union
import typing

from pydantic import BaseModel
import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool

from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.middleware_agent import create_agent
from langchain.tools.tool_node import ToolCallRequest, ToolCallResponse


class FakeModel(GenericFakeChatModel):
    """Fake chat model for testing."""

    tool_style: Literal["openai", "anthropic"] = "openai"

    def bind_tools(
        self,
        tools: typing.Sequence[Union[dict[str, Any], type[BaseModel], typing.Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        if len(tools) == 0:
            msg = "Must provide at least one tool"
            raise ValueError(msg)

        tool_dicts = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_dicts.append(tool)
                continue
            if not isinstance(tool, BaseTool):
                msg = "Only BaseTool and dict is supported by FakeModel.bind_tools"
                raise TypeError(msg)

            # NOTE: this is a simplified tool spec for testing purposes only
            if self.tool_style == "openai":
                tool_dicts.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                        },
                    }
                )
            elif self.tool_style == "anthropic":
                tool_dicts.append(
                    {
                        "name": tool.name,
                    }
                )

        return self.bind(tools=tool_dicts)


@tool
def add_tool(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


@tool
def failing_tool(x: int) -> int:
    """Tool that raises an error."""
    msg = "Intentional failure"
    raise ValueError(msg)


def test_single_middleware_on_tool_call():
    """Test that a single middleware can intercept tool calls."""
    call_log = []

    class LoggingMiddleware(AgentMiddleware):
        """Middleware that logs tool calls."""

        def on_tool_call(
            self, request: ToolCallRequest, state, runtime
        ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
            call_log.append(f"before_{request.tool.name}")
            response = yield request
            call_log.append(f"after_{request.tool.name}")
            return response

    model = FakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "add_tool", "args": {"x": 2, "y": 3}, "id": "1"}],
                ),
                AIMessage(content="Done"),
            ]
        )
    )

    agent = create_agent(
        model=model,
        tools=[add_tool],
        middleware=[LoggingMiddleware()],
    )

    result = agent.compile().invoke({"messages": [HumanMessage("Add 2 and 3")]})

    assert "before_add_tool" in call_log
    assert "after_add_tool" in call_log
    assert call_log.index("before_add_tool") < call_log.index("after_add_tool")

    # Check that tool executed successfully
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "5"


def test_multiple_middleware_chaining():
    """Test that multiple middleware chain correctly (outer wraps inner)."""
    call_order = []

    class OuterMiddleware(AgentMiddleware):
        """Outer middleware in the chain."""

        def on_tool_call(
            self, request: ToolCallRequest, state, runtime
        ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
            call_order.append("outer_start")
            response = yield request
            call_order.append("outer_end")
            return response

    class InnerMiddleware(AgentMiddleware):
        """Inner middleware in the chain."""

        def on_tool_call(
            self, request: ToolCallRequest, state, runtime
        ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
            call_order.append("inner_start")
            response = yield request
            call_order.append("inner_end")
            return response

    model = FakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "add_tool", "args": {"x": 1, "y": 1}, "id": "1"}],
                ),
                AIMessage(content="Done"),
            ]
        )
    )

    agent = create_agent(
        model=model,
        tools=[add_tool],
        middleware=[OuterMiddleware(), InnerMiddleware()],
    )

    agent.compile().invoke({"messages": [HumanMessage("Add 1 and 1")]})

    # Verify order: outer_start -> inner_start -> tool -> inner_end -> outer_end
    assert call_order == ["outer_start", "inner_start", "inner_end", "outer_end"]


def test_middleware_retry_logic():
    """Test that middleware can retry tool calls."""
    attempt_count = 0

    class RetryMiddleware(AgentMiddleware):
        """Middleware that retries on failure."""

        def on_tool_call(
            self, request: ToolCallRequest, state, runtime
        ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
            nonlocal attempt_count
            max_retries = 2

            for attempt in range(max_retries):
                attempt_count += 1
                response = yield request

                if response.action == "continue":
                    return response

                if response.action == "raise" and attempt < max_retries - 1:
                    # Retry
                    continue

                # Convert error to success message
                return ToolCallResponse(
                    action="continue",
                    result=ToolMessage(
                        content=f"Failed after {max_retries} attempts",
                        name=request.tool_call["name"],
                        tool_call_id=request.tool_call["id"],
                        status="error",
                    ),
                )

            raise AssertionError("Unreachable")

    model = FakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "failing_tool", "args": {"x": 1}, "id": "1"}],
                ),
                AIMessage(content="Done"),
            ]
        )
    )

    agent = create_agent(
        model=model,
        tools=[failing_tool],
        middleware=[RetryMiddleware()],
    )

    result = agent.compile().invoke({"messages": [HumanMessage("Test retry")]})

    # Should have attempted twice
    assert attempt_count == 2

    # Check that we got an error message
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Failed after 2 attempts" in tool_messages[0].content


def test_middleware_request_modification():
    """Test that middleware can modify tool requests."""

    class RequestModifierMiddleware(AgentMiddleware):
        """Middleware that doubles the input."""

        def on_tool_call(
            self, request: ToolCallRequest, state, runtime
        ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
            # Modify the arguments
            modified_tool_call = {
                **request.tool_call,
                "args": {
                    "x": request.tool_call["args"]["x"] * 2,
                    "y": request.tool_call["args"]["y"] * 2,
                },
            }
            modified_request = ToolCallRequest(
                tool_call=modified_tool_call,
                tool=request.tool,
            )
            response = yield modified_request
            return response

    model = FakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "add_tool", "args": {"x": 1, "y": 2}, "id": "1"}],
                ),
                AIMessage(content="Done"),
            ]
        )
    )

    agent = create_agent(
        model=model,
        tools=[add_tool],
        middleware=[RequestModifierMiddleware()],
    )

    result = agent.compile().invoke({"messages": [HumanMessage("Add 1 and 2")]})

    # Original: 1 + 2 = 3, Modified: 2 + 4 = 6
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "6"


def test_multiple_middleware_with_retry():
    """Test complex scenario with multiple middleware and retry logic."""
    call_log = []

    class MonitoringMiddleware(AgentMiddleware):
        """Outer middleware for monitoring."""

        def on_tool_call(
            self, request: ToolCallRequest, state, runtime
        ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
            call_log.append("monitoring_start")
            response = yield request
            call_log.append("monitoring_end")
            return response

    class RetryMiddleware(AgentMiddleware):
        """Inner middleware for retries."""

        def on_tool_call(
            self, request: ToolCallRequest, state, runtime
        ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
            call_log.append("retry_start")
            for attempt in range(2):
                call_log.append(f"retry_attempt_{attempt + 1}")
                response = yield request

                if response.action == "continue":
                    call_log.append("retry_success")
                    return response

                if attempt == 0:  # Retry once
                    call_log.append("retry_retry")
                    continue

            call_log.append("retry_failed")
            return response

    model = FakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "add_tool", "args": {"x": 5, "y": 7}, "id": "1"}],
                ),
                AIMessage(content="Done"),
            ]
        )
    )

    agent = create_agent(
        model=model,
        tools=[add_tool],
        middleware=[MonitoringMiddleware(), RetryMiddleware()],
    )

    agent.compile().invoke({"messages": [HumanMessage("Add 5 and 7")]})

    # Verify the call sequence
    assert call_log[0] == "monitoring_start"
    assert call_log[1] == "retry_start"
    assert "retry_attempt_1" in call_log
    assert "retry_success" in call_log
    assert call_log[-1] == "monitoring_end"


def test_mixed_middleware():
    """Test middleware with both before_model and on_tool_call hooks."""
    call_log = []

    class MixedMiddleware(AgentMiddleware):
        """Middleware with multiple hooks."""

        def on_tool_call(
            self, request: ToolCallRequest, state, runtime
        ) -> Generator[ToolCallRequest, ToolCallResponse, ToolCallResponse]:
            call_log.append("on_tool_call_start")
            for _ in range(3):
                response = yield request
                if response.action == "continue":
                    break
            # response = yield request
            call_log.append("on_tool_call_end")
            return response

    model = FakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "add_tool", "args": {"x": 10, "y": 20}, "id": "1"}],
                ),
                AIMessage(content="Done"),
            ]
        )
    )

    agent = create_agent(
        model=model,
        tools=[add_tool],
        middleware=[MixedMiddleware()],
    )

    agent.compile().invoke({"messages": [HumanMessage("Add 10 and 20")]})

    # Both hooks should have been called
    assert "before_model" in call_log
    assert "on_tool_call_start" in call_log
    assert "on_tool_call_end" in call_log
    # before_model runs before on_tool_call
    assert call_log.index("before_model") < call_log.index("on_tool_call_start")
