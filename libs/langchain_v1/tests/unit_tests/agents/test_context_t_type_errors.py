"""Demonstrate type errors that mypy catches for ContextT mismatches.

This file contains intentional type errors to demonstrate that mypy catches them.
Run: uv run --group lint mypy tests/unit_tests/agents/test_context_t_type_errors.py

Expected errors:
1. TypedDict "UserContext" has no key "session_id" - accessing wrong field
2. Argument incompatible with supertype - mismatched ModelRequest type
3. Cannot infer value of type parameter - middleware/context_schema mismatch
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class UserContext(TypedDict):
    user_id: str
    user_name: str


class SessionContext(TypedDict):
    session_id: str
    expires_at: int


# ERROR 1: Using wrong context fields
class WrongFieldsMiddleware(AgentMiddleware[AgentState[Any], UserContext]):
    def wrap_model_call(
        self,
        request: ModelRequest[UserContext],
        handler: Callable[[ModelRequest[UserContext]], ModelResponse],
    ) -> ModelResponse:
        # TYPE ERROR: 'session_id' doesn't exist on UserContext
        session_id: str = request.runtime.context["session_id"]
        _ = session_id
        return handler(request)


# ERROR 2: Mismatched ModelRequest type parameter in method signature
class MismatchedRequestMiddleware(AgentMiddleware[AgentState[Any], UserContext]):
    def wrap_model_call(
        self,
        # TYPE ERROR: Should be ModelRequest[UserContext], not SessionContext
        request: ModelRequest[SessionContext],
        handler: Callable[[ModelRequest[SessionContext]], ModelResponse],
    ) -> ModelResponse:
        return handler(request)


# ERROR 3: Middleware ContextT doesn't match context_schema
class SessionContextMiddleware(AgentMiddleware[AgentState[Any], SessionContext]):
    def wrap_model_call(
        self,
        request: ModelRequest[SessionContext],
        handler: Callable[[ModelRequest[SessionContext]], ModelResponse],
    ) -> ModelResponse:
        return handler(request)


def test_mismatched_context_schema() -> None:
    # TYPE ERROR: SessionContextMiddleware expects SessionContext,
    # but context_schema is UserContext
    _agent = create_agent(
        model="openai:gpt-4",
        middleware=[SessionContextMiddleware()],
        context_schema=UserContext,
    )


# ERROR 4: Backwards compatible middleware with typed context_schema
class BackwardsCompatibleMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(request)


def test_backwards_compat_with_context_schema() -> None:
    # TYPE ERROR: BackwardsCompatibleMiddleware is AgentMiddleware[..., None]
    # but context_schema=UserContext expects AgentMiddleware[..., UserContext]
    _agent = create_agent(
        model="openai:gpt-4",
        middleware=[BackwardsCompatibleMiddleware()],
        context_schema=UserContext,
    )
