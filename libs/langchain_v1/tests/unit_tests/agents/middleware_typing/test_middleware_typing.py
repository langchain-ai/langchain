"""Test file to verify type safety in middleware (ContextT and ResponseT).

This file demonstrates:
1. Backwards compatible middlewares (no type params specified) - works with defaults
2. Correctly typed middlewares (ContextT/ResponseT match) - full type safety
3. Type errors that are caught when types don't match

Run type check: uv run --group typing mypy <this file>
Run tests: uv run --group test pytest <this file> -v

To see type errors being caught, run:
  uv run --group typing mypy .../test_middleware_type_errors.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.graph.state import CompiledStateGraph
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.runtime import Runtime
    from langgraph.types import Command


# =============================================================================
# Context and Response schemas for testing
# =============================================================================
class UserContext(TypedDict):
    """Context with user information."""

    user_id: str
    user_name: str


class SessionContext(TypedDict):
    """Different context schema."""

    session_id: str
    expires_at: int


class AnalysisResult(BaseModel):
    """Structured response schema."""

    sentiment: str
    confidence: float


class SummaryResult(BaseModel):
    """Different structured response schema."""

    summary: str
    key_points: list[str]


# =============================================================================
# 1. BACKWARDS COMPATIBLE: Middlewares without type parameters
#    These work when create_agent has NO context_schema or response_format
# =============================================================================
class BackwardsCompatibleMiddleware(AgentMiddleware):
    """Middleware that doesn't specify type parameters - backwards compatible."""

    def before_model(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,  # No type param - backwards compatible!
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(request)


class BackwardsCompatibleMiddleware2(AgentMiddleware):
    """Another backwards compatible middleware using ModelRequest without params."""

    def wrap_model_call(
        self,
        request: ModelRequest,  # Unparameterized - defaults to ModelRequest[None]
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        _ = request.runtime
        return handler(request)


@before_model
def backwards_compatible_decorator(
    state: AgentState[Any], runtime: Runtime[Any]
) -> dict[str, Any] | None:
    """Decorator middleware without explicit type parameters."""
    return None


# =============================================================================
# 2. CORRECTLY TYPED: Middlewares with explicit ContextT
#    These work when create_agent has MATCHING context_schema
# =============================================================================
class UserContextMiddleware(AgentMiddleware[AgentState[Any], UserContext, Any]):
    """Middleware with correctly specified UserContext."""

    def before_model(
        self, state: AgentState[Any], runtime: Runtime[UserContext]
    ) -> dict[str, Any] | None:
        # Full type safety - IDE knows these fields exist
        _user_id: str = runtime.context["user_id"]
        _user_name: str = runtime.context["user_name"]
        return None

    def wrap_model_call(
        self,
        request: ModelRequest[UserContext],  # Correctly parameterized!
        handler: Callable[[ModelRequest[UserContext]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        # request.runtime.context is UserContext - fully typed!
        _user_id: str = request.runtime.context["user_id"]
        return handler(request)


class SessionContextMiddleware(AgentMiddleware[AgentState[Any], SessionContext, Any]):
    """Middleware with correctly specified SessionContext."""

    def wrap_model_call(
        self,
        request: ModelRequest[SessionContext],
        handler: Callable[[ModelRequest[SessionContext]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        _session_id: str = request.runtime.context["session_id"]
        _expires: int = request.runtime.context["expires_at"]
        return handler(request)


# =============================================================================
# 3. CORRECTLY TYPED: Middlewares with explicit ResponseT
#    These work when create_agent has MATCHING response_format
# =============================================================================
class AnalysisResponseMiddleware(
    AgentMiddleware[AgentState[AnalysisResult], ContextT, AnalysisResult]
):
    """Middleware with correctly specified AnalysisResult response type."""

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[AnalysisResult]],
    ) -> ModelResponse[AnalysisResult]:
        response = handler(request)
        # Full type safety on structured_response
        if response.structured_response is not None:
            _sentiment: str = response.structured_response.sentiment
            _confidence: float = response.structured_response.confidence
        return response


class SummaryResponseMiddleware(
    AgentMiddleware[AgentState[SummaryResult], ContextT, SummaryResult]
):
    """Middleware with correctly specified SummaryResult response type."""

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[SummaryResult]],
    ) -> ModelResponse[SummaryResult]:
        response = handler(request)
        if response.structured_response is not None:
            _summary: str = response.structured_response.summary
            _points: list[str] = response.structured_response.key_points
        return response


# =============================================================================
# 4. FULLY TYPED: Middlewares with both ContextT and ResponseT
# =============================================================================
class FullyTypedMiddleware(
    AgentMiddleware[AgentState[AnalysisResult], UserContext, AnalysisResult]
):
    """Middleware with both ContextT and ResponseT fully specified."""

    def wrap_model_call(
        self,
        request: ModelRequest[UserContext],
        handler: Callable[[ModelRequest[UserContext]], ModelResponse[AnalysisResult]],
    ) -> ModelResponse[AnalysisResult]:
        # Access context with full type safety
        _user_id: str = request.runtime.context["user_id"]

        response = handler(request)

        # Access structured response with full type safety
        if response.structured_response is not None:
            _sentiment: str = response.structured_response.sentiment

        return response


# =============================================================================
# 5. FLEXIBLE MIDDLEWARE: Works with any ContextT/ResponseT using Generic
# =============================================================================
class FlexibleMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Middleware that works with any ContextT and ResponseT."""

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        # Can't access specific fields, but works with any schemas
        _ = request.runtime
        return handler(request)


# =============================================================================
# 6. CREATE_AGENT INTEGRATION TESTS
# =============================================================================
@pytest.fixture
def fake_model() -> GenericFakeChatModel:
    """Create a fake model for testing."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))


def test_create_agent_no_context_schema(fake_model: GenericFakeChatModel) -> None:
    """Backwards compatible: No context_schema means ContextT=Any."""
    agent: CompiledStateGraph[Any, None, Any, Any] = create_agent(
        model=fake_model,
        middleware=[
            BackwardsCompatibleMiddleware(),
            BackwardsCompatibleMiddleware2(),
            backwards_compatible_decorator,
        ],
        # No context_schema - backwards compatible
    )
    assert agent is not None


def test_create_agent_with_user_context(fake_model: GenericFakeChatModel) -> None:
    """Typed: context_schema=UserContext requires matching middleware."""
    agent: CompiledStateGraph[Any, UserContext, Any, Any] = create_agent(
        model=fake_model,
        middleware=[UserContextMiddleware()],  # Matches UserContext
        context_schema=UserContext,
    )
    assert agent is not None


def test_create_agent_with_session_context(fake_model: GenericFakeChatModel) -> None:
    """Typed: context_schema=SessionContext requires matching middleware."""
    agent: CompiledStateGraph[Any, SessionContext, Any, Any] = create_agent(
        model=fake_model,
        middleware=[SessionContextMiddleware()],  # Matches SessionContext
        context_schema=SessionContext,
    )
    assert agent is not None


def test_create_agent_with_flexible_middleware(fake_model: GenericFakeChatModel) -> None:
    """Flexible middleware works with any context_schema."""
    # With UserContext
    agent1: CompiledStateGraph[Any, UserContext, Any, Any] = create_agent(
        model=fake_model,
        middleware=[FlexibleMiddleware[UserContext, Any]()],
        context_schema=UserContext,
    )
    assert agent1 is not None

    # With SessionContext
    agent2: CompiledStateGraph[Any, SessionContext, Any, Any] = create_agent(
        model=fake_model,
        middleware=[FlexibleMiddleware[SessionContext, Any]()],
        context_schema=SessionContext,
    )
    assert agent2 is not None


def test_create_agent_with_response_middleware(fake_model: GenericFakeChatModel) -> None:
    """Middleware with ResponseT works with response_format."""
    agent = create_agent(
        model=fake_model,
        middleware=[AnalysisResponseMiddleware()],
        response_format=AnalysisResult,
    )
    assert agent is not None


def test_create_agent_fully_typed(fake_model: GenericFakeChatModel) -> None:
    """Fully typed middleware with both ContextT and ResponseT."""
    agent = create_agent(
        model=fake_model,
        middleware=[FullyTypedMiddleware()],
        context_schema=UserContext,
        response_format=AnalysisResult,
    )
    assert agent is not None


# =============================================================================
# 7. ASYNC VARIANTS
# =============================================================================
class AsyncUserContextMiddleware(AgentMiddleware[AgentState[Any], UserContext, Any]):
    """Async middleware with correctly typed ContextT."""

    async def abefore_model(
        self, state: AgentState[Any], runtime: Runtime[UserContext]
    ) -> dict[str, Any] | None:
        _user_name: str = runtime.context["user_name"]
        return None

    async def awrap_model_call(
        self,
        request: ModelRequest[UserContext],
        handler: Callable[[ModelRequest[UserContext]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        _user_id: str = request.runtime.context["user_id"]
        return await handler(request)


class AsyncResponseMiddleware(
    AgentMiddleware[AgentState[AnalysisResult], ContextT, AnalysisResult]
):
    """Async middleware with correctly typed ResponseT."""

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[AnalysisResult]]],
    ) -> ModelResponse[AnalysisResult]:
        response = await handler(request)
        if response.structured_response is not None:
            _sentiment: str = response.structured_response.sentiment
        return response


def test_async_middleware_with_context(fake_model: GenericFakeChatModel) -> None:
    """Async middleware with typed context."""
    agent: CompiledStateGraph[Any, UserContext, Any, Any] = create_agent(
        model=fake_model,
        middleware=[AsyncUserContextMiddleware()],
        context_schema=UserContext,
    )
    assert agent is not None


def test_async_middleware_with_response(fake_model: GenericFakeChatModel) -> None:
    """Async middleware with typed response."""
    agent = create_agent(
        model=fake_model,
        middleware=[AsyncResponseMiddleware()],
        response_format=AnalysisResult,
    )
    assert agent is not None


# =============================================================================
# 8. MODEL_REQUEST AND MODEL_RESPONSE TESTS
# =============================================================================
def test_model_request_preserves_context_type() -> None:
    """Test that ModelRequest.override() preserves ContextT."""
    request: ModelRequest[UserContext] = ModelRequest(
        model=None,  # type: ignore[arg-type]
        messages=[HumanMessage(content="test")],
        runtime=None,
    )

    # Override should preserve the type parameter
    new_request: ModelRequest[UserContext] = request.override(
        messages=[HumanMessage(content="updated")]
    )

    assert type(request) is type(new_request)


def test_model_request_backwards_compatible() -> None:
    """Test that ModelRequest can be instantiated without type params."""
    request = ModelRequest(
        model=None,  # type: ignore[arg-type]
        messages=[HumanMessage(content="test")],
    )

    assert request.messages[0].content == "test"


def test_model_request_explicit_none() -> None:
    """Test ModelRequest[None] is same as unparameterized ModelRequest."""
    request1: ModelRequest[None] = ModelRequest(
        model=None,  # type: ignore[arg-type]
        messages=[HumanMessage(content="test")],
    )

    request2: ModelRequest = ModelRequest(
        model=None,  # type: ignore[arg-type]
        messages=[HumanMessage(content="test")],
    )

    assert type(request1) is type(request2)


def test_model_response_with_response_type() -> None:
    """Test that ModelResponse preserves ResponseT."""
    response: ModelResponse[AnalysisResult] = ModelResponse(
        result=[AIMessage(content="test")],
        structured_response=AnalysisResult(sentiment="positive", confidence=0.9),
    )

    # Type checker knows structured_response is AnalysisResult | None
    if response.structured_response is not None:
        _sentiment: str = response.structured_response.sentiment
        _confidence: float = response.structured_response.confidence


def test_model_response_without_structured() -> None:
    """Test ModelResponse without structured response."""
    response: ModelResponse[Any] = ModelResponse(
        result=[AIMessage(content="test")],
        structured_response=None,
    )

    assert response.structured_response is None


def test_model_response_backwards_compatible() -> None:
    """Test that ModelResponse can be instantiated without type params."""
    response = ModelResponse(
        result=[AIMessage(content="test")],
    )

    assert response.structured_response is None


# =============================================================================
# 9. ASYNC DECORATOR VARIANTS FOR wrap_model_call AND wrap_tool_call
# =============================================================================
@wrap_model_call
async def async_wrap_model_retry(
    request: ModelRequest[UserContext],
    handler: Callable[[ModelRequest[UserContext]], Awaitable[ModelResponse[Any]]],
) -> ModelResponse[Any] | AIMessage:
    """Async wrap_model_call decorator should type-check correctly."""
    for attempt in range(3):
        try:
            return await handler(request)
        except Exception:
            if attempt == 2:
                raise
    return await handler(request)


@wrap_model_call
async def async_wrap_model_unparameterized(
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse | AIMessage:
    """Async wrap_model_call with unparameterized types (backwards compat)."""
    return await handler(request)


@wrap_tool_call
async def async_wrap_tool_retry(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
) -> ToolMessage | Command[Any]:
    """Async wrap_tool_call decorator should type-check correctly."""
    for attempt in range(3):
        try:
            return await handler(request)
        except Exception:
            if attempt == 2:
                raise
    return await handler(request)


def test_async_wrap_model_call_decorator(fake_model: GenericFakeChatModel) -> None:
    """Async @wrap_model_call decorator produces valid AgentMiddleware."""
    agent = create_agent(
        model=fake_model,
        middleware=[async_wrap_model_retry],
        context_schema=UserContext,
    )
    assert agent is not None


def test_async_wrap_model_call_unparameterized(fake_model: GenericFakeChatModel) -> None:
    """Async @wrap_model_call with unparameterized types works."""
    agent = create_agent(
        model=fake_model,
        middleware=[async_wrap_model_unparameterized],
    )
    assert agent is not None


def test_async_wrap_tool_call_decorator(fake_model: GenericFakeChatModel) -> None:
    """Async @wrap_tool_call decorator produces valid AgentMiddleware."""
    agent = create_agent(
        model=fake_model,
        middleware=[async_wrap_tool_retry],
    )
    assert agent is not None


# Test sync decorators still work (regression check)
@wrap_model_call
def sync_wrap_model(
    request: ModelRequest[UserContext],
    handler: Callable[[ModelRequest[UserContext]], ModelResponse[Any]],
) -> ModelResponse[Any] | AIMessage:
    """Sync wrap_model_call should still type-check correctly."""
    return handler(request)


@wrap_tool_call
def sync_wrap_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:
    """Sync wrap_tool_call should still type-check correctly."""
    return handler(request)


def test_sync_wrap_model_call_decorator(fake_model: GenericFakeChatModel) -> None:
    """Sync @wrap_model_call decorator still works."""
    agent = create_agent(
        model=fake_model,
        middleware=[sync_wrap_model],
        context_schema=UserContext,
    )
    assert agent is not None


def test_sync_wrap_tool_call_decorator(fake_model: GenericFakeChatModel) -> None:
    """Sync @wrap_tool_call decorator still works."""
    agent = create_agent(
        model=fake_model,
        middleware=[sync_wrap_tool],
    )
    assert agent is not None


# Test with func=None pattern (parenthesized decorator)
@wrap_model_call()
async def async_wrap_model_with_parens(
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse | AIMessage:
    """Async wrap_model_call with parentheses should type-check correctly."""
    return await handler(request)


@wrap_tool_call()
async def async_wrap_tool_with_parens(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
) -> ToolMessage | Command[Any]:
    """Async wrap_tool_call with parentheses should type-check correctly."""
    return await handler(request)


def test_async_wrap_model_call_with_parens(fake_model: GenericFakeChatModel) -> None:
    """Async @wrap_model_call() with parens produces valid AgentMiddleware."""
    agent = create_agent(
        model=fake_model,
        middleware=[async_wrap_model_with_parens],
    )
    assert agent is not None


def test_async_wrap_tool_call_with_parens(fake_model: GenericFakeChatModel) -> None:
    """Async @wrap_tool_call() with parens produces valid AgentMiddleware."""
    agent = create_agent(
        model=fake_model,
        middleware=[async_wrap_tool_with_parens],
    )
    assert agent is not None
