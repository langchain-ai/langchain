"""Tests demonstrating proper typing support for middleware.

This test file verifies that:
1. ModelRequest is properly generic over StateT and ContextT
2. Async middleware decorators work without type errors
3. Sync middleware decorators work without type errors
4. Custom context types flow through properly
5. Handler callbacks have correct async/sync signatures

These tests should pass mypy type checking without any type: ignore comments.
"""

from collections.abc import Awaitable, Callable
from typing import TypedDict
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    after_agent,
    after_model,
    before_agent,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)


# Custom context type for testing
class ServiceContext(TypedDict):
    """Custom context for service-level information."""

    user_id: str
    session_id: str
    environment: str


class CustomState(AgentState):
    """Custom state extending AgentState."""

    custom_field: str


# =============================================================================
# Test 1: ModelRequest generic typing with custom context
# =============================================================================


def test_model_request_generic_context_typing() -> None:
    """Test that ModelRequest[StateT, ContextT] properly types the state and runtime fields."""
    # Create a mock model
    mock_model = MagicMock()

    # Create ModelRequest with explicit state and context type annotation
    request: ModelRequest[AgentState, ServiceContext] = ModelRequest(
        model=mock_model,
        messages=[HumanMessage(content="Hello")],
    )

    # The request should be created without type errors
    assert request.model == mock_model
    assert len(request.messages) == 1


# =============================================================================
# Test 2: Sync dynamic_prompt decorator with proper typing
# =============================================================================


def test_sync_dynamic_prompt_typing() -> None:
    """Test that sync @dynamic_prompt decorator works without type errors."""

    @dynamic_prompt
    def my_prompt(request: ModelRequest[AgentState, ServiceContext]) -> str:
        # This should work without type: ignore - accessing generic ModelRequest
        return f"System prompt for messages: {len(request.messages)}"

    # The decorator should return an AgentMiddleware
    assert isinstance(my_prompt, AgentMiddleware)


def test_sync_dynamic_prompt_returning_system_message() -> None:
    """Test that sync @dynamic_prompt can return SystemMessage."""

    @dynamic_prompt
    def my_prompt(request: ModelRequest[AgentState, None]) -> SystemMessage:
        return SystemMessage(content="You are a helpful assistant")

    assert isinstance(my_prompt, AgentMiddleware)


# =============================================================================
# Test 3: Async dynamic_prompt decorator with proper typing
# =============================================================================


def test_async_dynamic_prompt_typing() -> None:
    """Test that async @dynamic_prompt decorator works without type errors."""

    @dynamic_prompt
    async def my_async_prompt(request: ModelRequest[AgentState, ServiceContext]) -> str:
        # Async function should work without type errors
        return "Async system prompt"

    assert isinstance(my_async_prompt, AgentMiddleware)


def test_async_dynamic_prompt_returning_system_message() -> None:
    """Test that async @dynamic_prompt can return SystemMessage."""

    @dynamic_prompt
    async def my_async_prompt(request: ModelRequest[AgentState, None]) -> SystemMessage:
        return SystemMessage(content="Async system message")

    assert isinstance(my_async_prompt, AgentMiddleware)


# =============================================================================
# Test 4: Sync wrap_model_call decorator with proper handler typing
# =============================================================================


def test_sync_wrap_model_call_typing() -> None:
    """Test that sync @wrap_model_call decorator properly types the handler."""

    @wrap_model_call
    def retry_middleware(
        request: ModelRequest[AgentState, ServiceContext],
        handler: Callable[[ModelRequest[AgentState, ServiceContext]], ModelResponse],
    ) -> ModelCallResult:
        # Handler should be typed as sync - no Awaitable
        return handler(request)

    assert isinstance(retry_middleware, AgentMiddleware)


def test_sync_wrap_model_call_returning_ai_message() -> None:
    """Test that sync @wrap_model_call can return AIMessage directly."""

    @wrap_model_call
    def simple_middleware(
        request: ModelRequest[AgentState, None],
        handler: Callable[[ModelRequest[AgentState, None]], ModelResponse],
    ) -> ModelCallResult:
        # Can return AIMessage directly (converted automatically)
        return AIMessage(content="Simple response")

    assert isinstance(simple_middleware, AgentMiddleware)


# =============================================================================
# Test 5: Async wrap_model_call decorator with proper handler typing
# =============================================================================


def test_async_wrap_model_call_typing() -> None:
    """Test that async @wrap_model_call decorator properly types the async handler."""

    @wrap_model_call
    async def async_retry_middleware(
        request: ModelRequest[AgentState, ServiceContext],
        handler: Callable[[ModelRequest[AgentState, ServiceContext]], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        # Handler should be typed as async - returns Awaitable
        return await handler(request)

    assert isinstance(async_retry_middleware, AgentMiddleware)


def test_async_wrap_model_call_with_error_handling() -> None:
    """Test async @wrap_model_call with try/except pattern."""

    @wrap_model_call
    async def error_handling_middleware(
        request: ModelRequest[AgentState, None],
        handler: Callable[[ModelRequest[AgentState, None]], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        try:
            return await handler(request)
        except Exception:
            return AIMessage(content="Error occurred")

    assert isinstance(error_handling_middleware, AgentMiddleware)


# =============================================================================
# Test 6: Sync wrap_tool_call decorator with proper handler typing
# =============================================================================


def test_sync_wrap_tool_call_typing() -> None:
    """Test that sync @wrap_tool_call decorator properly types the handler."""

    @wrap_tool_call
    def tool_error_handler(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        try:
            return handler(request)
        except Exception as e:
            return ToolMessage(
                content=str(e),
                tool_call_id=request.tool_call["id"],
            )

    assert isinstance(tool_error_handler, AgentMiddleware)


# =============================================================================
# Test 7: Async wrap_tool_call decorator with proper handler typing
# =============================================================================


def test_async_wrap_tool_call_typing() -> None:
    """Test that async @wrap_tool_call decorator properly types the async handler."""

    @wrap_tool_call
    async def async_tool_error_handler(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        try:
            return await handler(request)
        except Exception as e:
            return ToolMessage(
                content=str(e),
                tool_call_id=request.tool_call["id"],
            )

    assert isinstance(async_tool_error_handler, AgentMiddleware)


# =============================================================================
# Test 8: before_model/after_model decorators with custom state
# =============================================================================


def test_before_model_with_custom_state_typing() -> None:
    """Test @before_model decorator with custom state schema."""

    @before_model(state_schema=CustomState)
    def log_before_model(
        state: CustomState,
        runtime: object,  # Runtime type comes from langgraph
    ) -> dict[str, object] | None:
        # Should have access to custom_field without type errors
        _ = state.get("custom_field")  # Access custom field
        return None

    assert isinstance(log_before_model, AgentMiddleware)
    assert log_before_model.state_schema == CustomState


def test_after_model_with_custom_state_typing() -> None:
    """Test @after_model decorator with custom state schema."""

    @after_model(state_schema=CustomState)
    def log_after_model(
        state: CustomState,
        runtime: object,
    ) -> dict[str, object] | None:
        return {"custom_field": "updated"}

    assert isinstance(log_after_model, AgentMiddleware)


# =============================================================================
# Test 9: before_agent/after_agent decorators
# =============================================================================


def test_before_agent_async_typing() -> None:
    """Test async @before_agent decorator."""

    @before_agent
    async def setup_agent(
        state: AgentState,
        runtime: object,
    ) -> dict[str, object] | None:
        return None

    assert isinstance(setup_agent, AgentMiddleware)


def test_after_agent_async_typing() -> None:
    """Test async @after_agent decorator."""

    @after_agent
    async def cleanup_agent(
        state: AgentState,
        runtime: object,
    ) -> dict[str, object] | None:
        return None

    assert isinstance(cleanup_agent, AgentMiddleware)


# =============================================================================
# Test 10: Class-based middleware with proper generic typing
# =============================================================================


class TypedMiddleware(AgentMiddleware[CustomState, ServiceContext]):
    """Class-based middleware with explicit type parameters."""

    state_schema = CustomState

    def before_model(
        self,
        state: CustomState,
        runtime: object,
    ) -> dict[str, object] | None:
        # State is properly typed as CustomState
        return None


def test_class_based_middleware_typing() -> None:
    """Test class-based middleware with explicit generics."""
    middleware = TypedMiddleware()
    assert middleware.state_schema == CustomState


# =============================================================================
# Test 11: ModelRequest.override() preserves generic type
# =============================================================================


def test_model_request_override_preserves_generic() -> None:
    """Test that ModelRequest.override() returns properly typed ModelRequest."""
    mock_model = MagicMock()

    request: ModelRequest[AgentState, ServiceContext] = ModelRequest(
        model=mock_model,
        messages=[HumanMessage(content="Hello")],
    )

    # override() should return ModelRequest[AgentState, ServiceContext], not ModelRequest[Any, Any]
    new_request = request.override(system_message=SystemMessage(content="New system prompt"))

    # This should be type-safe
    assert new_request.system_message is not None
    assert new_request.system_message.content == "New system prompt"


# =============================================================================
# Test 12: Multiple middleware in a list (simulating create_agent usage)
# =============================================================================


def test_middleware_list_typing() -> None:
    """Test that middleware can be collected in a properly typed list."""

    @dynamic_prompt
    async def system_prompt(request: ModelRequest[AgentState, ServiceContext]) -> SystemMessage:
        return SystemMessage(content="System")

    @wrap_model_call
    async def censor_response(
        request: ModelRequest[AgentState, ServiceContext],
        handler: Callable[[ModelRequest[AgentState, ServiceContext]], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(request)

    @wrap_tool_call
    async def handle_errors(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        try:
            return await handler(request)
        except Exception as e:
            return ToolMessage(content=str(e), tool_call_id=request.tool_call["id"])

    # All middleware should be assignable to a list of AgentMiddleware
    # Note: The decorators return AgentMiddleware with inferred generic parameters
    middleware_list: list[AgentMiddleware[AgentState, ServiceContext]] = [
        system_prompt,
        censor_response,
    ]

    assert len(middleware_list) == 2
