from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
)
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from collections.abc import Callable


class UserContext(TypedDict):
    user_id: str
    user_name: str


class SessionContext(TypedDict):
    session_id: str
    expires_at: int


class AnalysisResult(BaseModel):
    sentiment: str
    confidence: float


class SummaryResult(BaseModel):
    summary: str
    key_points: list[str]


class WrongContextFieldsMiddleware(AgentMiddleware[AgentState[Any], UserContext, Any]):
    def wrap_model_call(
        self,
        request: ModelRequest[UserContext],
        handler: Callable[[ModelRequest[UserContext]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        # TYPE ERROR: 'session_id' doesn't exist on UserContext
        session_id: str = request.runtime.context["session_id"]  # type: ignore[typeddict-item]
        _ = session_id
        return handler(request)


class MismatchedRequestMiddleware(AgentMiddleware[AgentState[Any], UserContext, Any]):
    def wrap_model_call(  # type: ignore[override]
        self,
        # TYPE ERROR: Should be ModelRequest[UserContext], not SessionContext
        request: ModelRequest[SessionContext],
        handler: Callable[[ModelRequest[SessionContext]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(request)


class SessionContextMiddleware(AgentMiddleware[AgentState[Any], SessionContext, Any]):
    def wrap_model_call(
        self,
        request: ModelRequest[SessionContext],
        handler: Callable[[ModelRequest[SessionContext]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(request)


def test_mismatched_context_schema() -> None:
    # TYPE ERROR: SessionContextMiddleware expects SessionContext,
    # but context_schema is UserContext
    fake_model = FakeToolCallingModel()
    _agent = create_agent(  # type: ignore[misc]
        model=fake_model,
        middleware=[SessionContextMiddleware()],
        context_schema=UserContext,
    )


class ExplicitNoneContextMiddleware(AgentMiddleware[AgentState[Any], None, Any]):
    """Middleware with explicit None context - should NOT work with context_schema."""

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(request)


def test_explicit_none_context_mismatch() -> None:
    # TYPE ERROR: ExplicitNoneContextMiddleware has ContextT=None,
    # but context_schema=UserContext expects AgentMiddleware[..., UserContext]
    fake_model = FakeToolCallingModel()
    _agent = create_agent(  # type: ignore[misc]
        model=fake_model,
        middleware=[ExplicitNoneContextMiddleware()],
        context_schema=UserContext,
    )


class WrongResponseFieldsMiddleware(
    AgentMiddleware[AgentState[AnalysisResult], ContextT, AnalysisResult]
):
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[AnalysisResult]],
    ) -> ModelResponse[AnalysisResult]:
        response = handler(request)
        # TYPE ERROR: 'summary' doesn't exist on AnalysisResult
        _summary: str = response.response.summary  # type: ignore[union-attr]
        return response


class UnparameterizedMiddleware(AgentMiddleware):
    """Unparameterized middleware - ContextT defaults to Any."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(request)


def test_unparameterized_middleware_with_context_schema() -> None:
    # NO TYPE ERROR: Unparameterized middleware defaults ContextT to Any,
    # so it's compatible with any context_schema
    fake_model = FakeToolCallingModel()
    _agent = create_agent(
        model=fake_model,
        middleware=[UnparameterizedMiddleware()],
        context_schema=UserContext,
    )
