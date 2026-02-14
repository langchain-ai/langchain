from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
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
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.graph.state import CompiledStateGraph
    from langgraph.runtime import Runtime


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


class BackwardsCompatibleMiddleware(AgentMiddleware):
    """Middleware that doesn't specify type parameters - defaults ContextT to Any."""

    def before_model(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,  # No type param - defaults to ModelRequest[Any]
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(request)


class BackwardsCompatibleMiddleware2(AgentMiddleware):
    """Another backwards compatible middleware using ModelRequest without params."""

    def wrap_model_call(
        self,
        request: ModelRequest,  # Unparameterized - defaults to ModelRequest[Any]
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


class UserContextMiddleware(AgentMiddleware[AgentState[Any], UserContext, Any]):
    """Middleware with correctly specified UserContext."""

    def before_model(
        self, state: AgentState[Any], runtime: Runtime[UserContext]
    ) -> dict[str, Any] | None:
        _user_id: str = runtime.context["user_id"]
        _user_name: str = runtime.context["user_name"]
        return None

    def wrap_model_call(
        self,
        request: ModelRequest[UserContext],
        handler: Callable[[ModelRequest[UserContext]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
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


class ExplicitNoneContextMiddleware(AgentMiddleware[AgentState[Any], None, Any]):
    """Middleware with explicit None context - should NOT work with context_schema."""

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(request)


def test_backwards_compatible_middleware_no_context() -> None:
    """Test that unparameterized middleware works without context_schema."""
    fake_model = GenericFakeChatModel(messages=AIMessage(content="Hello!"))
    agent = create_agent(
        model=fake_model,
        middleware=[BackwardsCompatibleMiddleware()],
    )
    assert agent is not None


def test_backwards_compatible_middleware_with_context() -> None:
    """Test that unparameterized middleware works WITH context_schema (ContextT=Any)."""
    fake_model = GenericFakeChatModel(messages=AIMessage(content="Hello!"))
    agent = create_agent(
        model=fake_model,
        middleware=[BackwardsCompatibleMiddleware()],
        context_schema=UserContext,
    )
    assert agent is not None


def test_typed_middleware_matching_context() -> None:
    """Test that typed middleware works when context_schema matches."""
    fake_model = GenericFakeChatModel(messages=AIMessage(content="Hello!"))
    agent = create_agent(
        model=fake_model,
        middleware=[UserContextMiddleware()],
        context_schema=UserContext,
    )
    assert agent is not None


def test_explicit_none_context_mismatch() -> None:
    """Test that explicit None context middleware should error with context_schema.
    
    This is the intentional mismatch case - middleware explicitly typed with
    ContextT=None should NOT be compatible with context_schema=UserContext.
    """
    fake_model = GenericFakeChatModel(messages=AIMessage(content="Hello!"))
    # This should raise a type error during type checking
    # The agent creation itself may succeed at runtime, but mypy should catch it
    _agent = create_agent(  # type: ignore[misc]
        model=fake_model,
        middleware=[ExplicitNoneContextMiddleware()],
        context_schema=UserContext,
    )
