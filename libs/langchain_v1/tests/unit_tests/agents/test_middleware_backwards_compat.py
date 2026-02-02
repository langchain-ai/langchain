"""Test backwards compatibility for middleware type parameters.

This file verifies that middlewares written BEFORE the ResponseT change still work.
All patterns that were valid before should remain valid.

Run type check: uv run --group lint mypy tests/unit_tests/agents/test_middleware_backwards_compat.py
Run tests: uv run --group test pytest tests/unit_tests/agents/test_middleware_backwards_compat.py -v
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

    from langgraph.runtime import Runtime


# =============================================================================
# OLD PATTERN 1: Completely unparameterized AgentMiddleware
# This was the most common pattern for simple middlewares
# =============================================================================
class OldStyleMiddleware1(AgentMiddleware):
    """Middleware with no type parameters at all - most common old pattern."""

    def before_model(self, state: AgentState[Any], runtime: Runtime[None]) -> dict[str, Any] | None:
        # Simple middleware that just logs or does something
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,  # No type param
        handler: Callable[[ModelRequest], ModelResponse],  # No type params
    ) -> ModelResponse:  # No type param
        return handler(request)


# =============================================================================
# OLD PATTERN 2: AgentMiddleware with only 2 type parameters (StateT, ContextT)
# This was the pattern before ResponseT was added
# =============================================================================
class OldStyleMiddleware2(AgentMiddleware[AgentState[Any], ContextT]):
    """Middleware with 2 type params - the old signature before ResponseT."""

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse],
    ) -> ModelResponse:
        return handler(request)


# =============================================================================
# OLD PATTERN 3: Middleware with explicit None context
# =============================================================================
class OldStyleMiddleware3(AgentMiddleware[AgentState[Any], None]):
    """Middleware explicitly typed for no context."""

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse],
    ) -> ModelResponse:
        return handler(request)


# =============================================================================
# OLD PATTERN 4: Middleware with specific context type (2 params)
# =============================================================================
class MyContext(TypedDict):
    user_id: str


class OldStyleMiddleware4(AgentMiddleware[AgentState[Any], MyContext]):
    """Middleware with specific context - old 2-param pattern."""

    def wrap_model_call(
        self,
        request: ModelRequest[MyContext],
        handler: Callable[[ModelRequest[MyContext]], ModelResponse],
    ) -> ModelResponse:
        # Access context fields
        _user_id: str = request.runtime.context["user_id"]
        return handler(request)


# =============================================================================
# OLD PATTERN 5: Decorator-based middleware
# =============================================================================
@before_model
def old_style_decorator(state: AgentState[Any], runtime: Runtime[None]) -> dict[str, Any] | None:
    """Decorator middleware - old pattern."""
    return None


# =============================================================================
# OLD PATTERN 6: Async middleware (2 params)
# =============================================================================
class OldStyleAsyncMiddleware(AgentMiddleware[AgentState[Any], ContextT]):
    """Async middleware with old 2-param pattern."""

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(request)


# =============================================================================
# OLD PATTERN 7: ModelResponse without type parameter
# =============================================================================
class OldStyleModelResponseMiddleware(AgentMiddleware):
    """Middleware using ModelResponse without type param."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        response = handler(request)
        # Access result - this always worked
        _ = response.result
        # structured_response was Any before, still works
        _ = response.structured_response
        return response


# =============================================================================
# TESTS: Verify all old patterns still work at runtime
# =============================================================================
@pytest.fixture
def fake_model() -> GenericFakeChatModel:
    """Create a fake model for testing."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))


def test_old_pattern_1_unparameterized(fake_model: GenericFakeChatModel) -> None:
    """Old pattern 1: Completely unparameterized middleware."""
    agent = create_agent(
        model=fake_model,
        middleware=[OldStyleMiddleware1()],
    )
    assert agent is not None


def test_old_pattern_2_two_params(fake_model: GenericFakeChatModel) -> None:
    """Old pattern 2: AgentMiddleware[StateT, ContextT] - 2 params."""
    agent = create_agent(
        model=fake_model,
        middleware=[OldStyleMiddleware2()],
    )
    assert agent is not None


def test_old_pattern_3_explicit_none(fake_model: GenericFakeChatModel) -> None:
    """Old pattern 3: Explicit None context."""
    agent = create_agent(
        model=fake_model,
        middleware=[OldStyleMiddleware3()],
    )
    assert agent is not None


def test_old_pattern_4_specific_context(fake_model: GenericFakeChatModel) -> None:
    """Old pattern 4: Specific context type with 2 params."""
    agent = create_agent(
        model=fake_model,
        middleware=[OldStyleMiddleware4()],
        context_schema=MyContext,
    )
    assert agent is not None


def test_old_pattern_5_decorator(fake_model: GenericFakeChatModel) -> None:
    """Old pattern 5: Decorator-based middleware."""
    agent = create_agent(
        model=fake_model,
        middleware=[old_style_decorator],
    )
    assert agent is not None


def test_old_pattern_6_async(fake_model: GenericFakeChatModel) -> None:
    """Old pattern 6: Async middleware with 2 params."""
    agent = create_agent(
        model=fake_model,
        middleware=[OldStyleAsyncMiddleware()],
    )
    assert agent is not None


def test_old_pattern_7_model_response_unparameterized(
    fake_model: GenericFakeChatModel,
) -> None:
    """Old pattern 7: ModelResponse without type parameter."""
    agent = create_agent(
        model=fake_model,
        middleware=[OldStyleModelResponseMiddleware()],
    )
    assert agent is not None


def test_multiple_old_style_middlewares(fake_model: GenericFakeChatModel) -> None:
    """Multiple old-style middlewares can be combined."""
    agent = create_agent(
        model=fake_model,
        middleware=[
            OldStyleMiddleware1(),
            OldStyleMiddleware2(),
            OldStyleMiddleware3(),
            old_style_decorator,
            OldStyleModelResponseMiddleware(),
        ],
    )
    assert agent is not None


def test_model_response_backwards_compat() -> None:
    """ModelResponse can be instantiated without type params."""
    # Old way - no type param
    response = ModelResponse(result=[AIMessage(content="test")])
    assert response.structured_response is None

    # Old way - accessing fields
    response2 = ModelResponse(
        result=[AIMessage(content="test")],
        structured_response={"key": "value"},
    )
    assert response2.structured_response == {"key": "value"}


def test_model_request_backwards_compat() -> None:
    """ModelRequest can be instantiated without type params."""
    # Old way - no type param
    request = ModelRequest(
        model=None,  # type: ignore[arg-type]
        messages=[HumanMessage(content="test")],
    )
    assert len(request.messages) == 1
