"""Test file to verify ContextT type safety in middleware.

This file demonstrates:
1. Backwards compatible middlewares (no ContextT specified) - works with no context_schema
2. Correctly typed middlewares (ContextT matches context_schema) - full type safety
3. Type errors that are caught when ContextT doesn't match

Run type check: uv run --group lint mypy tests/unit_tests/agents/test_context_t_typing.py
Run tests: uv run --group test pytest tests/unit_tests/agents/test_context_t_typing.py -v

To see type errors being caught, run:
  uv run --group lint mypy tests/unit_tests/agents/test_context_t_type_errors.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    before_model,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.graph.state import CompiledStateGraph
    from langgraph.runtime import Runtime


# =============================================================================
# Context schemas for testing
# =============================================================================
class UserContext(TypedDict):
    """Context with user information."""

    user_id: str
    user_name: str


class SessionContext(TypedDict):
    """Different context schema."""

    session_id: str
    expires_at: int


# =============================================================================
# 1. BACKWARDS COMPATIBLE: Middlewares without ContextT
#    These work when create_agent has NO context_schema specified
# =============================================================================
class BackwardsCompatibleMiddleware(AgentMiddleware):
    """Middleware that doesn't specify type parameters - backwards compatible."""

    def before_model(self, state: AgentState[Any], runtime: Runtime[None]) -> dict[str, Any] | None:
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
    state: AgentState[Any], runtime: Runtime[None]
) -> dict[str, Any] | None:
    """Decorator middleware without explicit ContextT."""
    return None


# =============================================================================
# 2. CORRECTLY TYPED: Middlewares with explicit ContextT
#    These work when create_agent has MATCHING context_schema
# =============================================================================
class UserContextMiddleware(AgentMiddleware[AgentState[Any], UserContext]):
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
        handler: Callable[[ModelRequest[UserContext]], ModelResponse],
    ) -> ModelResponse:
        # request.runtime.context is UserContext - fully typed!
        _user_id: str = request.runtime.context["user_id"]
        return handler(request)


class SessionContextMiddleware(AgentMiddleware[AgentState[Any], SessionContext]):
    """Middleware with correctly specified SessionContext."""

    def wrap_model_call(
        self,
        request: ModelRequest[SessionContext],
        handler: Callable[[ModelRequest[SessionContext]], ModelResponse],
    ) -> ModelResponse:
        _session_id: str = request.runtime.context["session_id"]
        _expires: int = request.runtime.context["expires_at"]
        return handler(request)


# =============================================================================
# 3. FLEXIBLE MIDDLEWARE: Works with any ContextT using Generic parameter
#    Use this pattern when middleware needs to work with different contexts
# =============================================================================
class FlexibleMiddleware(AgentMiddleware[AgentState[Any], ContextT]):
    """Middleware that works with any ContextT - uses the class's type parameter."""

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse],
    ) -> ModelResponse:
        # Can't access specific context fields, but works with any context_schema
        _ = request.runtime
        return handler(request)


# =============================================================================
# 4. CREATE_AGENT INTEGRATION TESTS
# =============================================================================
@pytest.fixture
def fake_model() -> GenericFakeChatModel:
    """Create a fake model for testing."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))


def test_create_agent_no_context_schema(fake_model: GenericFakeChatModel) -> None:
    """Backwards compatible: No context_schema means ContextT=None."""
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
        middleware=[FlexibleMiddleware[UserContext]()],
        context_schema=UserContext,
    )
    assert agent1 is not None

    # With SessionContext
    agent2: CompiledStateGraph[Any, SessionContext, Any, Any] = create_agent(
        model=fake_model,
        middleware=[FlexibleMiddleware[SessionContext]()],
        context_schema=SessionContext,
    )
    assert agent2 is not None


# =============================================================================
# 5. ASYNC VARIANTS
# =============================================================================
class AsyncUserContextMiddleware(AgentMiddleware[AgentState[Any], UserContext]):
    """Async middleware with correctly typed ContextT."""

    async def abefore_model(
        self, state: AgentState[Any], runtime: Runtime[UserContext]
    ) -> dict[str, Any] | None:
        _user_name: str = runtime.context["user_name"]
        return None

    async def awrap_model_call(
        self,
        request: ModelRequest[UserContext],
        handler: Callable[[ModelRequest[UserContext]], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        _user_id: str = request.runtime.context["user_id"]
        return await handler(request)


def test_async_middleware_with_context(fake_model: GenericFakeChatModel) -> None:
    """Async middleware with typed context."""
    agent: CompiledStateGraph[Any, UserContext, Any, Any] = create_agent(
        model=fake_model,
        middleware=[AsyncUserContextMiddleware()],
        context_schema=UserContext,
    )
    assert agent is not None


# =============================================================================
# 6. MODEL_REQUEST TESTS
# =============================================================================
def test_model_request_preserves_context_type() -> None:
    """Test that ModelRequest.override() preserves ContextT."""
    # Create a request with explicit type
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
    # This is the backwards compatible way - no type parameter
    request = ModelRequest(
        model=None,  # type: ignore[arg-type]
        messages=[HumanMessage(content="test")],
    )

    # The type is ModelRequest[None] due to ContextT default
    assert request.messages[0].content == "test"


def test_model_request_explicit_none() -> None:
    """Test ModelRequest[None] is same as unparameterized ModelRequest."""
    # Explicit None
    request1: ModelRequest[None] = ModelRequest(
        model=None,  # type: ignore[arg-type]
        messages=[HumanMessage(content="test")],
    )

    # Unparameterized (defaults to None)
    request2: ModelRequest = ModelRequest(
        model=None,  # type: ignore[arg-type]
        messages=[HumanMessage(content="test")],
    )

    # Both are the same type at runtime
    assert type(request1) is type(request2)
