from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)
from typing_extensions import TypeGuard

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain.agents.middleware.types import _InputAgentState, _OutputAgentState
    from langgraph.runtime import Runtime


StateT_co = TypeVar("StateT_co", bound="AgentState[Any]", covariant=True)
ContextT = TypeVar("ContextT", default=Any)
ResponseT = TypeVar("ResponseT", default=Any)


class AgentState(BaseModel, Generic[ResponseT]):
    """Base state for agent execution."""

    messages: list[AnyMessage]
    """The conversation history."""

    response: Optional[ResponseT] = None
    """The structured response, if any."""


class ModelRequest(BaseModel, Generic[ContextT]):
    """Request to call a model with context."""

    messages: list[AnyMessage]
    """The messages to send to the model."""

    tools: list[BaseTool]
    """The tools available to the model."""

    runtime: Runtime[ContextT]
    """The runtime context for the request."""

    config: RunnableConfig
    """The configuration for the request."""


class ModelResponse(BaseModel, Generic[ResponseT]):
    """Response from a model call."""

    message: AIMessage
    """The response message from the model."""

    response: Optional[ResponseT] = None
    """The structured response, if any."""


class ExtendedModelResponse(ModelResponse[ResponseT], Generic[ResponseT]):
    """Extended model response that can include commands."""

    commands: list[Command[Any]]
    """Commands to execute after the model call."""


class JumpTo(BaseModel):
    """Represents a jump to another node."""

    node: str
    """The node to jump to."""

    state: Optional[dict[str, Any]] = None
    """Optional state updates to apply."""


class OmitFromSchema:
    """Marker class to omit a field from the schema."""
    pass


class ToolCallRequest(BaseModel):
    """Request to call a tool."""

    tool_call: dict[str, Any]
    """The tool call to execute."""

    tool: Optional[BaseTool] = None
    """The tool to execute, if available."""


class ToolCallWrapper(BaseModel):
    """Wrapper for tool call execution."""

    tool_call: dict[str, Any]
    """The tool call being executed."""

    tool: BaseTool
    """The tool being executed."""


class AgentMiddleware(BaseModel, Generic[StateT_co, ContextT, ResponseT]):
    """Base class for agent middleware.

    Type parameters:
        StateT_co: The agent state type (covariant).
        ContextT: The context type for the runtime. Defaults to Any.
        ResponseT: The response type for structured output. Defaults to Any.
    """

    tools: list[BaseTool] = []
    """Tools provided by this middleware."""

    def before_model(
        self, state: StateT_co, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Called before the model is invoked.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Optional state updates to apply.
        """
        return None

    async def abefore_model(
        self, state: StateT_co, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async version of before_model."""
        return self.before_model(state, runtime)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Wrap the model call.

        Args:
            request: The model request.
            handler: The next handler in the chain.

        Returns:
            The model response.
        """
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Async version of wrap_model_call."""
        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolCallWrapper],
    ) -> ToolCallWrapper:
        """Wrap a tool call.

        Args:
            request: The tool call request.
            handler: The next handler in the chain.

        Returns:
            The tool call wrapper.
        """
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolCallWrapper]],
    ) -> ToolCallWrapper:
        """Async version of wrap_tool_call."""
        return await handler(request)


def before_model(
    func: Callable[[StateT_co, Runtime[ContextT]], dict[str, Any] | None]
) -> AgentMiddleware[StateT_co, ContextT, Any]:
    """Decorator to create middleware from a before_model function.

    Args:
        func: The before_model function.

    Returns:
        An AgentMiddleware instance.
    """

    class _BeforeModelMiddleware(AgentMiddleware[StateT_co, ContextT, Any]):
        def before_model(
            self, state: StateT_co, runtime: Runtime[ContextT]
        ) -> dict[str, Any] | None:
            return func(state, runtime)

    return _BeforeModelMiddleware()


@runtime_checkable
class _InputAgentState(Protocol[ResponseT]):
    """Protocol for input agent state."""

    messages: list[AnyMessage]


@runtime_checkable
class _OutputAgentState(Protocol[ResponseT]):
    """Protocol for output agent state."""

    messages: list[AnyMessage]
    response: Optional[ResponseT]
