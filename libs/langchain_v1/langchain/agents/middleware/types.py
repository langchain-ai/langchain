"""Types for middleware and agents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from inspect import iscoroutinefunction
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    Protocol,
    cast,
    overload,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

# Needed as top level import for Pydantic schema generation on AgentState
import warnings
from typing import TypeAlias

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolCallWrapper
from langgraph.typing import ContextT
from typing_extensions import NotRequired, Required, TypedDict, TypeVar, Unpack

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime
    from langgraph.types import Command

    from langchain.agents.structured_output import ResponseFormat

__all__ = [
    "AgentMiddleware",
    "AgentState",
    "ContextT",
    "ExtendedModelResponse",
    "ModelCallResult",
    "ModelRequest",
    "ModelResponse",
    "OmitFromSchema",
    "ResponseT",
    "StateT_co",
    "ToolCallRequest",
    "ToolCallWrapper",
    "after_agent",
    "after_model",
    "before_agent",
    "before_model",
    "dynamic_prompt",
    "hook_config",
    "wrap_tool_call",
]

JumpTo = Literal["tools", "model", "end"]
"""Destination to jump to when a middleware node returns."""

ResponseT = TypeVar("ResponseT", default=Any)


class _ModelRequestOverrides(TypedDict, total=False):
    """Possible overrides for `ModelRequest.override()` method."""

    model: BaseChatModel
    system_message: SystemMessage | None
    messages: list[AnyMessage]
    tool_choice: Any | None
    tools: list[BaseTool | dict[str, Any]]
    response_format: ResponseFormat[Any] | None
    model_settings: dict[str, Any]
    state: AgentState[Any]


@dataclass(init=False)
class ModelRequest(Generic[ContextT]):
    """Model request information for the agent.

    Type Parameters:
        ContextT: The type of the runtime context. Defaults to `None` if not specified.
    """

    model: BaseChatModel
    messages: list[AnyMessage]  # excluding system message
    system_message: SystemMessage | None
    tool_choice: Any | None
    tools: list[BaseTool | dict[str, Any]]
    response_format: ResponseFormat[Any] | None
    state: AgentState[Any]
    runtime: Runtime[ContextT]
    model_settings: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        model: BaseChatModel,
        messages: list[AnyMessage],
        system_message: SystemMessage | None = None,
        system_prompt: str | None = None,
        tool_choice: Any | None = None,
        tools: list[BaseTool | dict[str, Any]] | None = None,
        response_format: ResponseFormat[Any] | None = None,
        state: AgentState[Any] | None = None,
        runtime: Runtime[ContextT] | None = None,
        model_settings: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ModelRequest with backward compatibility for system_prompt.

        Args:
            model: The chat model to use.
            messages: List of messages (excluding system prompt).
            tool_choice: Tool choice configuration.
            tools: List of available tools.
            response_format: Response format specification.
            state: Agent state.
            runtime: Runtime context.
            model_settings: Additional model settings.
            system_message: System message instance (preferred).
            system_prompt: System prompt string (deprecated, converted to SystemMessage).

        Raises:
            ValueError: If both `system_prompt` and `system_message` are provided.
        """
        # Handle system_prompt/system_message conversion and validation
        if system_prompt is not None and system_message is not None:
            msg = "Cannot specify both system_prompt and system_message"
            raise ValueError(msg)

        if system_prompt is not None:
            system_message = SystemMessage(content=system_prompt)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            self.model = model
            self.messages = messages
            self.system_message = system_message
            self.tool_choice = tool_choice
            self.tools = tools if tools is not None else []
            self.response_format = response_format
            self.state = state if state is not None else {"messages": []}
            self.runtime = runtime  # type: ignore[assignment]
            self.model_settings = model_settings if model_settings is not None else {}

    @property
    def system_prompt(self) -> str | None:
        """Get system prompt text from system_message.

        Returns:
            The content of the system message if present, otherwise `None`.
        """
        if self.system_message is None:
            return None
        return self.system_message.text

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute with a deprecation warning.

        Direct attribute assignment on `ModelRequest` is deprecated. Use the
        `override()` method instead to create a new request with modified attributes.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        # Special handling for system_prompt - convert to system_message
        if name == "system_prompt":
            warnings.warn(
                "Direct attribute assignment to ModelRequest.system_prompt is deprecated. "
                "Use request.override(system_message=SystemMessage(...)) instead to create "
                "a new request with the modified system message.",
                DeprecationWarning,
                stacklevel=2,
            )
            if value is None:
                object.__setattr__(self, "system_message", None)
            else:
                object.__setattr__(self, "system_message", SystemMessage(content=value))
            return

        warnings.warn(
            f"Direct attribute assignment to ModelRequest.{name} is deprecated. "
            f"Use request.override({name}=...) instead to create a new request "
            f"with the modified attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        object.__setattr__(self, name, value)

    def override(self, **overrides: Unpack[_ModelRequestOverrides]) -> ModelRequest[ContextT]:
        """Replace the request with a new request with the given overrides.

        Returns a new `ModelRequest` instance with the specified attributes replaced.

        This follows an immutable pattern, leaving the original request unchanged.

        Args:
            **overrides: Keyword arguments for attributes to override.

                Supported keys:

                - `model`: `BaseChatModel` instance
                - `system_prompt`: deprecated, use `system_message` instead
                - `system_message`: `SystemMessage` instance
                - `messages`: `list` of messages
                - `tool_choice`: Tool choice configuration
                - `tools`: `list` of available tools
                - `response_format`: Response format specification
                - `model_settings`: Additional model settings
                - `state`: Agent state dictionary

        Returns:
            New `ModelRequest` instance with specified overrides applied.

        Examples:
            !!! example "Create a new request with different model"

                ```python
                new_request = request.override(model=different_model)
                ```

            !!! example "Override system message (preferred)"

                ```python
                from langchain_core.messages import SystemMessage

                new_request = request.override(
                    system_message=SystemMessage(content="New instructions")
                )
                ```

            !!! example "Override multiple attributes"

                ```python
                new_request = request.override(
                    model=ChatOpenAI(model="gpt-4o"),
                    system_message=SystemMessage(content="New instructions"),
                )
                ```

        Raises:
            ValueError: If both `system_prompt` and `system_message` are provided.
        """
        # Handle system_prompt/system_message conversion
        if "system_prompt" in overrides and "system_message" in overrides:
            msg = "Cannot specify both system_prompt and system_message"
            raise ValueError(msg)

        if "system_prompt" in overrides:
            system_prompt = cast("str | None", overrides.pop("system_prompt"))  # type: ignore[typeddict-item]
            if system_prompt is None:
                overrides["system_message"] = None
            else:
                overrides["system_message"] = SystemMessage(content=system_prompt)

        return replace(self, **overrides)


@dataclass
class ModelResponse(Generic[ResponseT]):
    """Response from model execution including messages and optional structured output.

    The result will usually contain a single `AIMessage`, but may include an additional
    `ToolMessage` if the model used a tool for structured output.

    Type Parameters:
        ResponseT: The type of the structured response. Defaults to `Any` if not specified.
    """

    result: list[BaseMessage]
    """List of messages from model execution."""

    structured_response: ResponseT | None = None
    """Parsed structured output if `response_format` was specified, `None` otherwise."""


@dataclass
class ExtendedModelResponse(Generic[ResponseT]):
    """Model response with an optional 'Command' from 'wrap_model_call' middleware.

    Use this to return a 'Command' alongside the model response from a
    'wrap_model_call' handler. The command is applied as an additional state
    update after the model node completes, using the graph's reducers (e.g.
    'add_messages' for the 'messages' key).

    Because each 'Command' is applied through the reducer, messages in the
    command are **added alongside** the model response messages rather than
    replacing them. For non-reducer state fields, later commands overwrite
    earlier ones (outermost middleware wins over inner).

    Type Parameters:
        ResponseT: The type of the structured response. Defaults to 'Any' if not specified.
    """

    model_response: ModelResponse[ResponseT]
    """The underlying model response."""

    command: Command[Any] | None = None
    """Optional command to apply as an additional state update."""


ModelCallResult: TypeAlias = (
    "ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]"
)
"""`TypeAlias` for model call handler return value.

Middleware can return either:

- `ModelResponse`: Full response with messages and optional structured output
- `AIMessage`: Simplified return for simple use cases
- `ExtendedModelResponse`: Response with an optional `Command` for additional state updates
    `goto`, `resume`, and `graph` are not yet supported on these commands.
    A `NotImplementedError` will be raised if you try to use them.
"""


@dataclass
class OmitFromSchema:
    """Annotation used to mark state attributes as omitted from input or output schemas."""

    input: bool = True
    """Whether to omit the attribute from the input schema."""

    output: bool = True
    """Whether to omit the attribute from the output schema."""


OmitFromInput = OmitFromSchema(input=True, output=False)
"""Annotation used to mark state attributes as omitted from input schema."""

OmitFromOutput = OmitFromSchema(input=False, output=True)
"""Annotation used to mark state attributes as omitted from output schema."""

PrivateStateAttr = OmitFromSchema(input=True, output=True)
"""Annotation used to mark state attributes as purely internal for a given middleware."""


class AgentState(TypedDict, Generic[ResponseT]):
    """State schema for the agent."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]


class _InputAgentState(TypedDict):  # noqa: PYI049
    """Input state schema for the agent."""

    messages: Required[Annotated[list[AnyMessage | dict[str, Any]], add_messages]]


class _OutputAgentState(TypedDict, Generic[ResponseT]):  # noqa: PYI049
    """Output state schema for the agent."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    structured_response: NotRequired[ResponseT]


StateT = TypeVar("StateT", bound=AgentState[Any], default=AgentState[Any])
StateT_co = TypeVar("StateT_co", bound=AgentState[Any], default=AgentState[Any], covariant=True)
StateT_contra = TypeVar("StateT_contra", bound=AgentState[Any], contravariant=True)


class _DefaultAgentState(AgentState[Any]):
    """AgentMiddleware default state."""


class AgentMiddleware(Generic[StateT, ContextT, ResponseT]):
    """Base middleware class for an agent.

    Subclass this and implement any of the defined methods to customize agent behavior
    between steps in the main agent loop.

    Type Parameters:
        StateT: The type of the agent state. Defaults to `AgentState[Any]`.
        ContextT: The type of the runtime context. Defaults to `None`.
        ResponseT: The type of the structured response. Defaults to `Any`.
    """

    state_schema: type[StateT] = cast("type[StateT]", _DefaultAgentState)
    """The schema for state passed to the middleware nodes."""

    tools: Sequence[BaseTool]
    """Additional tools registered by the middleware."""

    @property
    def name(self) -> str:
        """The name of the middleware instance.

        Defaults to the class name, but can be overridden for custom naming.
        """
        return self.__class__.__name__

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the agent execution starts.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply before agent execution.
        """

    async def abefore_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run before the agent execution starts.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply before agent execution.
        """

    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the model is called.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply before model call.
        """

    async def abefore_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run before the model is called.

        Args:
            state: The agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply before model call.
        """

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the model is called.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply after model call.
        """

    async def aafter_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run after the model is called.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply after model call.
        """

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Intercept and control model execution via handler callback.

        Async version is `awrap_model_call`

        The handler callback executes the model request and returns a `ModelResponse`.
        Middleware can call the handler multiple times for retry logic, skip calling
        it to short-circuit, or modify the request/response. Multiple middleware
        compose with first in list as outermost layer.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Callback that executes the model request and returns
                `ModelResponse`.

                Call this to execute the model.

                Can be called multiple times for retry logic.

                Can skip calling it to short-circuit.

        Returns:
            The model call result.

        Examples:
            !!! example "Retry on error"

                ```python
                def wrap_model_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            return handler(request)
                        except Exception:
                            if attempt == 2:
                                raise
                ```

            !!! example "Rewrite response"

                ```python
                def wrap_model_call(self, request, handler):
                    response = handler(request)
                    ai_msg = response.result[0]
                    return ModelResponse(
                        result=[AIMessage(content=f"[{ai_msg.content}]")],
                        structured_response=response.structured_response,
                    )
                ```

            !!! example "Error to fallback"

                ```python
                def wrap_model_call(self, request, handler):
                    try:
                        return handler(request)
                    except Exception:
                        return ModelResponse(result=[AIMessage(content="Service unavailable")])
                ```

            !!! example "Cache/short-circuit"

                ```python
                def wrap_model_call(self, request, handler):
                    if cached := get_cache(request):
                        return cached  # Short-circuit with cached result
                    response = handler(request)
                    save_cache(request, response)
                    return response
                ```

            !!! example "Simple `AIMessage` return (converted automatically)"

                ```python
                def wrap_model_call(self, request, handler):
                    response = handler(request)
                    # Can return AIMessage directly for simple cases
                    return AIMessage(content="Simplified response")
                ```
        """
        msg = (
            "Synchronous implementation of wrap_model_call is not available. "
            "You are likely encountering this error because you defined only the async version "
            "(awrap_model_call) and invoked your agent in a synchronous context "
            "(e.g., using `stream()` or `invoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the synchronous wrap_model_call method, "
            "(2) use the @wrap_model_call decorator on a standalone sync function, or "
            "(3) invoke your agent asynchronously using `astream()` or `ainvoke()`."
        )
        raise NotImplementedError(msg)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Intercept and control async model execution via handler callback.

        The handler callback executes the model request and returns a `ModelResponse`.

        Middleware can call the handler multiple times for retry logic, skip calling
        it to short-circuit, or modify the request/response. Multiple middleware
        compose with first in list as outermost layer.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Async callback that executes the model request and returns
                `ModelResponse`.

                Call this to execute the model.

                Can be called multiple times for retry logic.

                Can skip calling it to short-circuit.

        Returns:
            The model call result.

        Examples:
            !!! example "Retry on error"

                ```python
                async def awrap_model_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            return await handler(request)
                        except Exception:
                            if attempt == 2:
                                raise
                ```
        """
        msg = (
            "Asynchronous implementation of awrap_model_call is not available. "
            "You are likely encountering this error because you defined only the sync version "
            "(wrap_model_call) and invoked your agent in an asynchronous context "
            "(e.g., using `astream()` or `ainvoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the asynchronous awrap_model_call method, "
            "(2) use the @wrap_model_call decorator on a standalone async function, or "
            "(3) invoke your agent synchronously using `stream()` or `invoke()`."
        )
        raise NotImplementedError(msg)

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the agent execution completes.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply after agent execution.
        """

    async def aafter_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run after the agent execution completes.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Agent state updates to apply after agent execution.
        """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Intercept tool execution for retries, monitoring, or modification.

        Async version is `awrap_tool_call`

        Multiple middleware compose automatically (first defined = outermost).

        Exceptions propagate unless `handle_tool_errors` is configured on `ToolNode`.

        Args:
            request: Tool call request with call `dict`, `BaseTool`, state, and runtime.

                Access state via `request.state` and runtime via `request.runtime`.
            handler: `Callable` to execute the tool (can be called multiple times).

        Returns:
            `ToolMessage` or `Command` (the final result).

        The handler `Callable` can be invoked multiple times for retry logic.

        Each call to handler is independent and stateless.

        Examples:
            !!! example "Modify request before execution"

                ```python
                def wrap_tool_call(self, request, handler):
                    modified_call = {
                        **request.tool_call,
                        "args": {
                            **request.tool_call["args"],
                            "value": request.tool_call["args"]["value"] * 2,
                        },
                    }
                    request = request.override(tool_call=modified_call)
                    return handler(request)
                ```

            !!! example "Retry on error (call handler multiple times)"

                ```python
                def wrap_tool_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            result = handler(request)
                            if is_valid(result):
                                return result
                        except Exception:
                            if attempt == 2:
                                raise
                    return result
                ```

            !!! example "Conditional retry based on response"

                ```python
                def wrap_tool_call(self, request, handler):
                    for attempt in range(3):
                        result = handler(request)
                        if isinstance(result, ToolMessage) and result.status != "error":
                            return result
                        if attempt < 2:
                            continue
                        return result
                ```
        """
        msg = (
            "Synchronous implementation of wrap_tool_call is not available. "
            "You are likely encountering this error because you defined only the async version "
            "(awrap_tool_call) and invoked your agent in a synchronous context "
            "(e.g., using `stream()` or `invoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the synchronous wrap_tool_call method, "
            "(2) use the @wrap_tool_call decorator on a standalone sync function, or "
            "(3) invoke your agent asynchronously using `astream()` or `ainvoke()`."
        )
        raise NotImplementedError(msg)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Intercept and control async tool execution via handler callback.

        The handler callback executes the tool call and returns a `ToolMessage` or
        `Command`. Middleware can call the handler multiple times for retry logic, skip
        calling it to short-circuit, or modify the request/response. Multiple middleware
        compose with first in list as outermost layer.

        Args:
            request: Tool call request with call `dict`, `BaseTool`, state, and runtime.

                Access state via `request.state` and runtime via `request.runtime`.
            handler: Async callable to execute the tool and returns `ToolMessage` or
                `Command`.

                Call this to execute the tool.

                Can be called multiple times for retry logic.

                Can skip calling it to short-circuit.

        Returns:
            `ToolMessage` or `Command` (the final result).

        The handler `Callable` can be invoked multiple times for retry logic.

        Each call to handler is independent and stateless.

        Examples:
            !!! example "Async retry on error"

                ```python
                async def awrap_tool_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            result = await handler(request)
                            if is_valid(result):
                                return result
                        except Exception:
                            if attempt == 2:
                                raise
                    return result
                ```

                ```python
                async def awrap_tool_call(self, request, handler):
                    if cached := await get_cache_async(request):
                        return ToolMessage(content=cached, tool_call_id=request.tool_call["id"])
                    result = await handler(request)
                    await save_cache_async(request, result)
                    return result
                ```
        """
        msg = (
            "Asynchronous implementation of awrap_tool_call is not available. "
            "You are likely encountering this error because you defined only the sync version "
            "(wrap_tool_call) and invoked your agent in an asynchronous context "
            "(e.g., using `astream()` or `ainvoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the asynchronous awrap_tool_call method, "
            "(2) use the @wrap_tool_call decorator on a standalone async function, or "
            "(3) invoke your agent synchronously using `stream()` or `invoke()`."
        )
        raise NotImplementedError(msg)


class _CallableWithStateAndRuntime(Protocol[StateT_contra, ContextT]):
    """Callable with `AgentState` and `Runtime` as arguments."""

    def __call__(
        self, state: StateT_contra, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | Command[Any] | None | Awaitable[dict[str, Any] | Command[Any] | None]:
        """Perform some logic with the state and runtime."""
        ...


class _CallableReturningSystemMessage(Protocol[StateT_contra, ContextT]):  # type: ignore[misc]
    """Callable that returns a prompt string or SystemMessage given `ModelRequest`."""

    def __call__(
        self, request: ModelRequest[ContextT]
    ) -> str | SystemMessage | Awaitable[str | SystemMessage]:
        """Generate a system prompt string or SystemMessage based on the request."""
        ...


class _CallableReturningModelResponse(Protocol[StateT_contra, ContextT, ResponseT]):  # type: ignore[misc]
    """Callable for model call interception with handler callback.

    Receives handler callback to execute model and returns `ModelResponse` or
    `AIMessage`.
    """

    def __call__(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Intercept model execution via handler callback."""
        ...


class _CallableReturningToolResponse(Protocol):
    """Callable for tool call interception with handler callback.

    Receives handler callback to execute tool and returns final `ToolMessage` or
    `Command`.
    """

    def __call__(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Intercept tool execution via handler callback."""
        ...


CallableT = TypeVar("CallableT", bound=Callable[..., Any])


def hook_config(
    *,
    can_jump_to: list[JumpTo] | None = None,
) -> Callable[[CallableT], CallableT]:
    """Decorator to configure hook behavior in middleware methods.

    Use this decorator on `before_model` or `after_model` methods in middleware classes
    to configure their behavior. Currently supports specifying which destinations they
    can jump to, which establishes conditional edges in the agent graph.

    Args:
        can_jump_to: Optional list of valid jump destinations.

            Can be:

            - `'tools'`: Jump to the tools node
            - `'model'`: Jump back to the model node
            - `'end'`: Jump to the end of the graph

    Returns:
        Decorator function that marks the method with configuration metadata.

    Examples:
        !!! example "Using decorator on a class method"

            ```python
            class MyMiddleware(AgentMiddleware):
                @hook_config(can_jump_to=["end", "model"])
                def before_model(self, state: AgentState) -> dict[str, Any] | None:
                    if some_condition(state):
                        return {"jump_to": "end"}
                    return None
            ```

        Alternative: Use the `can_jump_to` parameter in `before_model`/`after_model`
        decorators:

        ```python
        @before_model(can_jump_to=["end"])
        def conditional_middleware(state: AgentState) -> dict[str, Any] | None:
            if should_exit(state):
                return {"jump_to": "end"}
            return None
        ```
    """

    def decorator(func: CallableT) -> CallableT:
        if can_jump_to is not None:
            func.__can_jump_to__ = can_jump_to  # type: ignore[attr-defined]
        return func

    return decorator


@overload
def before_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def before_model(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def before_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the `before_model` hook.

    Args:
        func: The function to be decorated.

            Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime
                context
        state_schema: Optional custom state schema type.

            If not provided, uses the default `AgentState` schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.

            Valid values are: `'tools'`, `'model'`, `'end'`
        name: Optional name for the generated middleware class.

            If not provided, uses the decorated function's name.

    Returns:
        Either an `AgentMiddleware` instance (if func is provided directly) or a
            decorator function that can be applied to a function it is wrapping.

    The decorated function should return:

    - `dict[str, Any]` - State updates to merge into the agent state
    - `Command` - A command to control flow (e.g., jump to different node)
    - `None` - No state updates or flow control

    Examples:
        !!! example "Basic usage"

            ```python
            @before_model
            def log_before_model(state: AgentState, runtime: Runtime) -> None:
                print(f"About to call model with {len(state['messages'])} messages")
            ```

        !!! example "With conditional jumping"

            ```python
            @before_model(can_jump_to=["end"])
            def conditional_before_model(
                state: AgentState, runtime: Runtime
            ) -> dict[str, Any] | None:
                if some_condition(state):
                    return {"jump_to": "end"}
                return None
            ```

        !!! example "With custom state schema"

            ```python
            @before_model(state_schema=MyCustomState)
            def custom_before_model(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "updated_value"}
            ```

        !!! example "Streaming custom events before model call"

            Use `runtime.stream_writer` to emit custom events before each model invocation.
            Events are received when streaming with `stream_mode="custom"`.

            ```python
            @before_model
            async def notify_model_call(state: AgentState, runtime: Runtime) -> None:
                '''Notify user before model is called.'''
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Thinking...",
                    }
                )
            ```
    """

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command[Any] | None:
                return await func(state, runtime)  # type: ignore[misc]

            # Preserve can_jump_to metadata on the wrapped function
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "BeforeModelMiddleware")
            )

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "abefore_model": async_wrapped,
                },
            )()

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command[Any] | None:
            return func(state, runtime)  # type: ignore[return-value]

        # Preserve can_jump_to metadata on the wrapped function
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

        # Use function name as default if no name provided
        middleware_name = name or cast("str", getattr(func, "__name__", "BeforeModelMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "before_model": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def after_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def after_model(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def after_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the `after_model` hook.

    Args:
        func: The function to be decorated.

            Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime
            context
        state_schema: Optional custom state schema type.

            If not provided, uses the default `AgentState` schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.

            Valid values are: `'tools'`, `'model'`, `'end'`
        name: Optional name for the generated middleware class.

            If not provided, uses the decorated function's name.

    Returns:
        Either an `AgentMiddleware` instance (if func is provided) or a decorator
            function that can be applied to a function.

    The decorated function should return:

    - `dict[str, Any]` - State updates to merge into the agent state
    - `Command` - A command to control flow (e.g., jump to different node)
    - `None` - No state updates or flow control

    Examples:
        !!! example "Basic usage for logging model responses"

            ```python
            @after_model
            def log_latest_message(state: AgentState, runtime: Runtime) -> None:
                print(state["messages"][-1].content)
            ```

        !!! example "With custom state schema"

            ```python
            @after_model(state_schema=MyCustomState, name="MyAfterModelMiddleware")
            def custom_after_model(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "updated_after_model"}
            ```

        !!! example "Streaming custom events after model call"

            Use `runtime.stream_writer` to emit custom events after model responds.
            Events are received when streaming with `stream_mode="custom"`.

            ```python
            @after_model
            async def notify_model_response(state: AgentState, runtime: Runtime) -> None:
                '''Notify user after model has responded.'''
                last_message = state["messages"][-1]
                has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Using tools..." if has_tool_calls else "Response ready!",
                    }
                )
            ```
    """

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)
        # Extract can_jump_to from decorator parameter or from function metadata
        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command[Any] | None:
                return await func(state, runtime)  # type: ignore[misc]

            # Preserve can_jump_to metadata on the wrapped function
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast("str", getattr(func, "__name__", "AfterModelMiddleware"))

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "aafter_model": async_wrapped,
                },
            )()

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command[Any] | None:
            return func(state, runtime)  # type: ignore[return-value]

        # Preserve can_jump_to metadata on the wrapped function
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

        # Use function name as default if no name provided
        middleware_name = name or cast("str", getattr(func, "__name__", "AfterModelMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "after_model": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def before_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def before_agent(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def before_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the `before_agent` hook.

    Args:
        func: The function to be decorated.

            Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime
            context
        state_schema: Optional custom state schema type.

            If not provided, uses the default `AgentState` schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.

            Valid values are: `'tools'`, `'model'`, `'end'`
        name: Optional name for the generated middleware class.

            If not provided, uses the decorated function's name.

    Returns:
        Either an `AgentMiddleware` instance (if func is provided directly) or a
            decorator function that can be applied to a function it is wrapping.

    The decorated function should return:

    - `dict[str, Any]` - State updates to merge into the agent state
    - `Command` - A command to control flow (e.g., jump to different node)
    - `None` - No state updates or flow control

    Examples:
        !!! example "Basic usage"

            ```python
            @before_agent
            def log_before_agent(state: AgentState, runtime: Runtime) -> None:
                print(f"Starting agent with {len(state['messages'])} messages")
            ```

        !!! example "With conditional jumping"

            ```python
            @before_agent(can_jump_to=["end"])
            def conditional_before_agent(
                state: AgentState, runtime: Runtime
            ) -> dict[str, Any] | None:
                if some_condition(state):
                    return {"jump_to": "end"}
                return None
            ```

        !!! example "With custom state schema"

            ```python
            @before_agent(state_schema=MyCustomState)
            def custom_before_agent(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "initialized_value"}
            ```

        !!! example "Streaming custom events"

            Use `runtime.stream_writer` to emit custom events during agent execution.
            Events are received when streaming with `stream_mode="custom"`.

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import before_agent, AgentState
            from langchain.messages import HumanMessage
            from langgraph.runtime import Runtime


            @before_agent
            async def notify_start(state: AgentState, runtime: Runtime) -> None:
                '''Notify user that agent is starting.'''
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Initializing agent session...",
                    }
                )
                # Perform prerequisite tasks here
                runtime.stream_writer({"type": "status", "message": "Agent ready!"})


            agent = create_agent(
                model="openai:gpt-5.2",
                tools=[...],
                middleware=[notify_start],
            )

            # Consume with stream_mode="custom" to receive events
            async for mode, event in agent.astream(
                {"messages": [HumanMessage("Hello")]},
                stream_mode=["updates", "custom"],
            ):
                if mode == "custom":
                    print(f"Status: {event}")
            ```
    """

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command[Any] | None:
                return await func(state, runtime)  # type: ignore[misc]

            # Preserve can_jump_to metadata on the wrapped function
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "BeforeAgentMiddleware")
            )

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "abefore_agent": async_wrapped,
                },
            )()

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command[Any] | None:
            return func(state, runtime)  # type: ignore[return-value]

        # Preserve can_jump_to metadata on the wrapped function
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

        # Use function name as default if no name provided
        middleware_name = name or cast("str", getattr(func, "__name__", "BeforeAgentMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "before_agent": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def after_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def after_agent(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def after_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the `after_agent` hook.

    Async version is `aafter_agent`.

    Args:
        func: The function to be decorated.

            Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime
            context
        state_schema: Optional custom state schema type.

            If not provided, uses the default `AgentState` schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.

            Valid values are: `'tools'`, `'model'`, `'end'`
        name: Optional name for the generated middleware class.

            If not provided, uses the decorated function's name.

    Returns:
        Either an `AgentMiddleware` instance (if func is provided) or a decorator
            function that can be applied to a function.

    The decorated function should return:

    - `dict[str, Any]` - State updates to merge into the agent state
    - `Command` - A command to control flow (e.g., jump to different node)
    - `None` - No state updates or flow control

    Examples:
        !!! example "Basic usage for logging agent completion"

            ```python
            @after_agent
            def log_completion(state: AgentState, runtime: Runtime) -> None:
                print(f"Agent completed with {len(state['messages'])} messages")
            ```

        !!! example "With custom state schema"

            ```python
            @after_agent(state_schema=MyCustomState, name="MyAfterAgentMiddleware")
            def custom_after_agent(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "finalized_value"}
            ```

        !!! example "Streaming custom events on completion"

            Use `runtime.stream_writer` to emit custom events when agent completes.
            Events are received when streaming with `stream_mode="custom"`.

            ```python
            @after_agent
            async def notify_completion(state: AgentState, runtime: Runtime) -> None:
                '''Notify user that agent has completed.'''
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Agent execution complete!",
                        "total_messages": len(state["messages"]),
                    }
                )
            ```
    """

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)
        # Extract can_jump_to from decorator parameter or from function metadata
        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command[Any] | None:
                return await func(state, runtime)  # type: ignore[misc]

            # Preserve can_jump_to metadata on the wrapped function
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast("str", getattr(func, "__name__", "AfterAgentMiddleware"))

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "aafter_agent": async_wrapped,
                },
            )()

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command[Any] | None:
            return func(state, runtime)  # type: ignore[return-value]

        # Preserve can_jump_to metadata on the wrapped function
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

        # Use function name as default if no name provided
        middleware_name = name or cast("str", getattr(func, "__name__", "AfterAgentMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "after_agent": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def dynamic_prompt(
    func: _CallableReturningSystemMessage[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def dynamic_prompt(
    func: None = None,
) -> Callable[
    [_CallableReturningSystemMessage[StateT, ContextT]],
    AgentMiddleware[StateT, ContextT],
]: ...


def dynamic_prompt(
    func: _CallableReturningSystemMessage[StateT, ContextT] | None = None,
) -> (
    Callable[
        [_CallableReturningSystemMessage[StateT, ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically generate system prompts for the model.

    This is a convenience decorator that creates middleware using `wrap_model_call`
    specifically for dynamic prompt generation. The decorated function should return
    a string that will be set as the system prompt for the model request.

    Args:
        func: The function to be decorated.

            Must accept: `request: ModelRequest` - Model request (contains state and
            runtime)

    Returns:
        Either an `AgentMiddleware` instance (if func is provided) or a decorator
            function that can be applied to a function.

    The decorated function should return:
        - `str` – The system prompt string to use for the model request
        - `SystemMessage` – A complete system message to use for the model request

    Examples:
        Basic usage with dynamic content:

        ```python
        @dynamic_prompt
        def my_prompt(request: ModelRequest) -> str:
            user_name = request.runtime.context.get("user_name", "User")
            return f"You are a helpful assistant helping {user_name}."
        ```

        Using state to customize the prompt:

        ```python
        @dynamic_prompt
        def context_aware_prompt(request: ModelRequest) -> str:
            msg_count = len(request.state["messages"])
            if msg_count > 10:
                return "You are in a long conversation. Be concise."
            return "You are a helpful assistant."
        ```

        Using with agent:

        ```python
        agent = create_agent(model, middleware=[my_prompt])
        ```
    """

    def decorator(
        func: _CallableReturningSystemMessage[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                request: ModelRequest[ContextT],
                handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[Any]]],
            ) -> ModelResponse[Any] | AIMessage:
                prompt = await func(request)  # type: ignore[misc]
                if isinstance(prompt, SystemMessage):
                    request = request.override(system_message=prompt)
                else:
                    request = request.override(system_message=SystemMessage(content=prompt))
                return await handler(request)

            middleware_name = cast("str", getattr(func, "__name__", "DynamicPromptMiddleware"))

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": AgentState,
                    "tools": [],
                    "awrap_model_call": async_wrapped,
                },
            )()

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], ModelResponse[Any]],
        ) -> ModelResponse[Any] | AIMessage:
            prompt = cast("Callable[[ModelRequest[ContextT]], SystemMessage | str]", func)(request)
            if isinstance(prompt, SystemMessage):
                request = request.override(system_message=prompt)
            else:
                request = request.override(system_message=SystemMessage(content=prompt))
            return handler(request)

        async def async_wrapped_from_sync(
            _self: AgentMiddleware[StateT, ContextT],
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[Any]]],
        ) -> ModelResponse[Any] | AIMessage:
            # Delegate to sync function
            prompt = cast("Callable[[ModelRequest[ContextT]], SystemMessage | str]", func)(request)
            if isinstance(prompt, SystemMessage):
                request = request.override(system_message=prompt)
            else:
                request = request.override(system_message=SystemMessage(content=prompt))
            return await handler(request)

        middleware_name = cast("str", getattr(func, "__name__", "DynamicPromptMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": AgentState,
                "tools": [],
                "wrap_model_call": wrapped,
                "awrap_model_call": async_wrapped_from_sync,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def wrap_model_call(
    func: _CallableReturningModelResponse[StateT, ContextT, ResponseT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def wrap_model_call(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableReturningModelResponse[StateT, ContextT, ResponseT]],
    AgentMiddleware[StateT, ContextT],
]: ...


def wrap_model_call(
    func: _CallableReturningModelResponse[StateT, ContextT, ResponseT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableReturningModelResponse[StateT, ContextT, ResponseT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
):
    """Create middleware with `wrap_model_call` hook from a function.

    Converts a function with handler callback into middleware that can intercept model
    calls, implement retry logic, handle errors, and rewrite responses.

    Args:
        func: Function accepting (request, handler) that calls handler(request)
            to execute the model and returns `ModelResponse` or `AIMessage`.

            Request contains state and runtime.
        state_schema: Custom state schema.

            Defaults to `AgentState`.
        tools: Additional tools to register with this middleware.
        name: Middleware class name.

            Defaults to function name.

    Returns:
        `AgentMiddleware` instance if func provided, otherwise a decorator.

    Examples:
        !!! example "Basic retry logic"

            ```python
            @wrap_model_call
            def retry_on_error(request, handler):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        return handler(request)
                    except Exception:
                        if attempt == max_retries - 1:
                            raise
            ```

        !!! example "Model fallback"

            ```python
            @wrap_model_call
            def fallback_model(request, handler):
                # Try primary model
                try:
                    return handler(request)
                except Exception:
                    pass

                # Try fallback model
                request = request.override(model=fallback_model_instance)
                return handler(request)
            ```

        !!! example "Rewrite response content (full `ModelResponse`)"

            ```python
            @wrap_model_call
            def uppercase_responses(request, handler):
                response = handler(request)
                ai_msg = response.result[0]
                return ModelResponse(
                    result=[AIMessage(content=ai_msg.content.upper())],
                    structured_response=response.structured_response,
                )
            ```

        !!! example "Simple `AIMessage` return (converted automatically)"

            ```python
            @wrap_model_call
            def simple_response(request, handler):
                # AIMessage is automatically converted to ModelResponse
                return AIMessage(content="Simple response")
            ```
    """

    def decorator(
        func: _CallableReturningModelResponse[StateT, ContextT, ResponseT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                request: ModelRequest[ContextT],
                handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
            ) -> ModelResponse[ResponseT] | AIMessage:
                return await func(request, handler)  # type: ignore[misc, arg-type]

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "WrapModelCallMiddleware")
            )

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "awrap_model_call": async_wrapped,
                },
            )()

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
        ) -> ModelResponse[ResponseT] | AIMessage:
            return func(request, handler)

        middleware_name = name or cast("str", getattr(func, "__name__", "WrapModelCallMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "wrap_model_call": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def wrap_tool_call(
    func: _CallableReturningToolResponse,
) -> AgentMiddleware: ...


@overload
def wrap_tool_call(
    func: None = None,
    *,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableReturningToolResponse],
    AgentMiddleware,
]: ...


def wrap_tool_call(
    func: _CallableReturningToolResponse | None = None,
    *,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableReturningToolResponse],
        AgentMiddleware,
    ]
    | AgentMiddleware
):
    """Create middleware with `wrap_tool_call` hook from a function.

    Async version is `awrap_tool_call`.

    Converts a function with handler callback into middleware that can intercept
    tool calls, implement retry logic, monitor execution, and modify responses.

    Args:
        func: Function accepting (request, handler) that calls
            handler(request) to execute the tool and returns final `ToolMessage` or
            `Command`.

            Can be sync or async.
        tools: Additional tools to register with this middleware.
        name: Middleware class name.

            Defaults to function name.

    Returns:
        `AgentMiddleware` instance if func provided, otherwise a decorator.

    Examples:
        !!! example "Retry logic"

            ```python
            @wrap_tool_call
            def retry_on_error(request, handler):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        return handler(request)
                    except Exception:
                        if attempt == max_retries - 1:
                            raise
            ```

        !!! example "Async retry logic"

            ```python
            @wrap_tool_call
            async def async_retry(request, handler):
                for attempt in range(3):
                    try:
                        return await handler(request)
                    except Exception:
                        if attempt == 2:
                            raise
            ```

        !!! example "Modify request"

            ```python
            @wrap_tool_call
            def modify_args(request, handler):
                modified_call = {
                    **request.tool_call,
                    "args": {
                        **request.tool_call["args"],
                        "value": request.tool_call["args"]["value"] * 2,
                    },
                }
                request = request.override(tool_call=modified_call)
                return handler(request)
            ```

        !!! example "Short-circuit with cached result"

            ```python
            @wrap_tool_call
            def with_cache(request, handler):
                if cached := get_cache(request):
                    return ToolMessage(content=cached, tool_call_id=request.tool_call["id"])
                result = handler(request)
                save_cache(request, result)
                return result
            ```
    """

    def decorator(
        func: _CallableReturningToolResponse,
    ) -> AgentMiddleware:
        is_async = iscoroutinefunction(func)

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware,
                request: ToolCallRequest,
                handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
            ) -> ToolMessage | Command[Any]:
                return await func(request, handler)  # type: ignore[arg-type,misc]

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "WrapToolCallMiddleware")
            )

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": AgentState,
                    "tools": tools or [],
                    "awrap_tool_call": async_wrapped,
                },
            )()

        def wrapped(
            _self: AgentMiddleware,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
        ) -> ToolMessage | Command[Any]:
            return func(request, handler)

        middleware_name = name or cast("str", getattr(func, "__name__", "WrapToolCallMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": AgentState,
                "tools": tools or [],
                "wrap_tool_call": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator
