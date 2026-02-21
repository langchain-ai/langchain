"""Demonstrate type errors that mypy catches for ContextT and ResponseT mismatches.

This file contains intentional type errors to demonstrate that mypy catches them.
Run: uv run --group typing mypy <this file>

Expected errors:
1. TypedDict "UserContext" has no key "session_id" - accessing wrong context field
2. Argument incompatible with supertype - mismatched ModelRequest type
3. Cannot infer value of type parameter - middleware/context_schema mismatch
4. "AnalysisResult" has no attribute "summary" - accessing wrong response field
5. Handler returns wrong ResponseT type
"""

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


# =============================================================================
# Context and Response schemas
# =============================================================================
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


# =============================================================================
# ERROR 1: Using wrong context fields
# =============================================================================
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


# =============================================================================
# ERROR 2: Mismatched ModelRequest type parameter in method signature
# =============================================================================
class MismatchedRequestMiddleware(AgentMiddleware[AgentState[Any], UserContext, Any]):
    def wrap_model_call(  # type: ignore[override]
        self,
        # TYPE ERROR: Should be ModelRequest[UserContext], not SessionContext
        request: ModelRequest[SessionContext],
        handler: Callable[[ModelRequest[SessionContext]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(request)


# =============================================================================
# ERROR 3: Middleware ContextT doesn't match context_schema
# =============================================================================
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


# =============================================================================
# ERROR 4: Backwards compatible middleware with typed context_schema
# =============================================================================
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
    fake_model = FakeToolCallingModel()
    _agent = create_agent(  # type: ignore[misc]
        model=fake_model,
        middleware=[BackwardsCompatibleMiddleware()],
        context_schema=UserContext,
    )


# =============================================================================
# ERROR 5: Using wrong response fields
# =============================================================================
class WrongResponseFieldsMiddleware(
    AgentMiddleware[AgentState[AnalysisResult], ContextT, AnalysisResult]
):
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[AnalysisResult]],
    ) -> ModelResponse[AnalysisResult]:
        response = handler(request)
        if response.structured_response is not None:
            # TYPE ERROR: 'summary' doesn't exist on AnalysisResult
            summary: str = response.structured_response.summary  # type: ignore[attr-defined]
            _ = summary
        return response


# =============================================================================
# ERROR 6: Mismatched ResponseT in method signature
# =============================================================================
class MismatchedResponseMiddleware(
    AgentMiddleware[AgentState[AnalysisResult], ContextT, AnalysisResult]
):
    def wrap_model_call(  # type: ignore[override]
        self,
        request: ModelRequest[ContextT],
        # TYPE ERROR: Handler should return ModelResponse[AnalysisResult], not SummaryResult
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[SummaryResult]],
    ) -> ModelResponse[AnalysisResult]:
        # This would fail at runtime - types don't match
        return handler(request)  # type: ignore[return-value]


# =============================================================================
# ERROR 7: Middleware ResponseT doesn't match response_format
# =============================================================================
class AnalysisMiddleware(AgentMiddleware[AgentState[AnalysisResult], ContextT, AnalysisResult]):
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[AnalysisResult]],
    ) -> ModelResponse[AnalysisResult]:
        return handler(request)


def test_mismatched_response_format() -> None:
    # TODO: TYPE ERROR not yet detected by mypy - AnalysisMiddleware expects AnalysisResult,
    # but response_format is SummaryResult. This requires more sophisticated typing.
    fake_model = FakeToolCallingModel()
    _agent = create_agent(
        model=fake_model,
        middleware=[AnalysisMiddleware()],
        response_format=SummaryResult,
    )


# =============================================================================
# ERROR 8: Wrong return type from wrap_model_call
# =============================================================================
class WrongReturnTypeMiddleware(
    AgentMiddleware[AgentState[AnalysisResult], ContextT, AnalysisResult]
):
    def wrap_model_call(  # type: ignore[override]
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[AnalysisResult]],
    ) -> ModelResponse[SummaryResult]:  # TYPE ERROR: Should return ModelResponse[AnalysisResult]
        return handler(request)  # type: ignore[return-value]
