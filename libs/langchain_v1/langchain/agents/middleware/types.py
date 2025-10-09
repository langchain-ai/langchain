"""Types for middleware and agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
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

    from langchain.tools.tool_node import ToolCallRequest

# needed as top level import for pydantic schema generation on AgentState
from langchain_core.messages import AnyMessage, BaseMessage, ToolMessage  # noqa: TC002
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.graph.message import add_messages
from langgraph.types import Command  # noqa: TC002
from langgraph.typing import ContextT
from typing_extensions import NotRequired, Required, TypedDict, TypeVar

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime

    from langchain.agents.structured_output import ResponseFormat

__all__ = [
    "AgentMiddleware",
    "AgentState",
    "ContextT",
    "ModelCall",
    "ModelRequest",
    "ModelResponse",
    "OmitFromSchema",
    "PublicAgentState",
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

ResponseT = TypeVar("ResponseT")


@dataclass
class ModelCall:
    """Model invocation parameters for a single model call.

    Contains only the parameters needed to invoke the model, without agent context.
    """

    model: BaseChatModel
    system_prompt: str | None
    messages: list[AnyMessage]  # excluding system prompt
    tool_choice: Any | None
    tools: list[BaseTool | dict]
    response_format: ResponseFormat | None
    model_settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRequest:
    """Full request context for model invocation including agent state.

    Combines model invocation parameters with agent state and runtime context.
    """

    model_call: ModelCall
    state: AgentState
    runtime: Runtime[ContextT]  # type: ignore[valid-type]


@dataclass
class ModelResponse:
    """Response from model execution including messages and optional structured output.

    The result will usually contain a single AIMessage, but may include
    an additional ToolMessage if the model used a tool for structured output.
    """

    result: list[BaseMessage]
    """List of messages from model execution."""

    structured_response: Any = None
    """Parsed structured output if response_format was specified, None otherwise."""


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
    thread_model_call_count: NotRequired[Annotated[int, PrivateStateAttr]]
    run_model_call_count: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]


class PublicAgentState(TypedDict, Generic[ResponseT]):
    """Public state schema for the agent.

    Just used for typing purposes.
    """

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    structured_response: NotRequired[ResponseT]


StateT = TypeVar("StateT", bound=AgentState, default=AgentState)
StateT_contra = TypeVar("StateT_contra", bound=AgentState, contravariant=True)


class AgentMiddleware(Generic[StateT, ContextT]):
    """Base middleware class for an agent.

    Subclass this and implement any of the defined methods to customize agent behavior
    between steps in the main agent loop.
    """

    state_schema: type[StateT] = cast("type[StateT]", AgentState)
    """The schema for state passed to the middleware nodes."""

    tools: list[BaseTool]
    """Additional tools registered by the middleware."""

    @property
    def name(self) -> str:
        """The name of the middleware instance.

        Defaults to the class name, but can be overridden for custom naming.
        """
        return self.__class__.__name__

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the agent execution starts."""

    async def abefore_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run before the agent execution starts."""

    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the model is called."""

    async def abefore_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run before the model is called."""

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the model is called."""

    async def aafter_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run after the model is called."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelCall], ModelResponse],
    ) -> ModelResponse:
        """Intercept and control model execution via handler callback.

        The handler callback executes the model call and returns a ModelResponse containing
        messages and optional structured_response. Middleware can call the handler multiple
        times for retry logic, skip calling it to short-circuit, or modify the request/response.
        Multiple middleware compose with first in list as outermost layer.

        Args:
            request: Full model request including state and runtime context.
            handler: Callback that executes the model call and returns ModelResponse.
                     Pass request.model_call to execute the model. Can be called
                     multiple times for retry logic. Can skip calling it to short-circuit.

        Returns:
            Final ModelResponse to use (from handler or custom).

        Examples:
            Retry on error:
            ```python
            def wrap_model_call(self, request, handler):
                for attempt in range(3):
                    try:
                        return handler(request.model_call)
                    except Exception:
                        if attempt == 2:
                            raise
            ```

            Modify messages:
            ```python
            def wrap_model_call(self, request, handler):
                response = handler(request.model_call)
                # Modify first message (AIMessage)
                ai_msg = response.result[0]
                modified = AIMessage(content=f"[{ai_msg.content}]")
                return ModelResponse(
                    result=[modified, *response.result[1:]],
                    structured_response=response.structured_response,
                )
            ```

            Error to fallback:
            ```python
            def wrap_model_call(self, request, handler):
                try:
                    return handler(request.model_call)
                except Exception:
                    return ModelResponse(result=[AIMessage(content="Service unavailable")])
            ```

            Modify model settings:
            ```python
            def wrap_model_call(self, request, handler):
                # Modify the model call parameters
                request.model_call.model_settings["temperature"] = 0.7
                return handler(request.model_call)
            ```
        """
        raise NotImplementedError

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelCall], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call.

        Args:
            request: Full model request including state and runtime context.
            handler: Async callback that executes the model call and returns ModelResponse.
                     Pass request.model_call to execute the model.

        Returns:
            Final ModelResponse to use (from handler or custom).

        Examples:
            Retry on error:
            ```python
            async def awrap_model_call(self, request, handler):
                for attempt in range(3):
                    try:
                        return await handler(request.model_call)
                    except Exception:
                        if attempt == 2:
                            raise
            ```
        """
        raise NotImplementedError

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the agent execution completes."""

    async def aafter_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run after the agent execution completes."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool execution for retries, monitoring, or modification.

        Multiple middleware compose automatically (first defined = outermost).
        Exceptions propagate unless handle_tool_errors is configured on ToolNode.

        Args:
            request: Tool call request with call dict, BaseTool, state, and runtime.
                Access state via request.state and runtime via request.runtime.
            handler: Callable to execute the tool (can be called multiple times).

        Returns:
            ToolMessage or Command (the final result).

        The handler callable can be invoked multiple times for retry logic.
        Each call to handler is independent and stateless.

        Examples:
            Modify request before execution:

            def wrap_tool_call(self, request, handler):
                request.tool_call["args"]["value"] *= 2
                return handler(request)

            Retry on error (call handler multiple times):

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

            Conditional retry based on response:

            def wrap_tool_call(self, request, handler):
                for attempt in range(3):
                    result = handler(request)
                    if isinstance(result, ToolMessage) and result.status != "error":
                        return result
                    if attempt < 2:
                        continue
                    return result
        """
        raise NotImplementedError


class _CallableWithStateAndRuntime(Protocol[StateT_contra, ContextT]):
    """Callable with AgentState and Runtime as arguments."""

    def __call__(
        self, state: StateT_contra, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | Command | None | Awaitable[dict[str, Any] | Command | None]:
        """Perform some logic with the state and runtime."""
        ...


class _CallableReturningPromptString(Protocol[StateT_contra, ContextT]):  # type: ignore[misc]
    """Callable that returns a prompt string given ModelRequest (contains state and runtime)."""

    def __call__(self, request: ModelRequest) -> str | Awaitable[str]:
        """Generate a system prompt string based on the request."""
        ...


class _CallableReturningModelResponse(Protocol[StateT_contra, ContextT]):  # type: ignore[misc]
    """Callable for model call interception with handler callback.

    Receives handler callback to execute model and returns final ModelResponse.
    """

    def __call__(
        self,
        request: ModelRequest,
        handler: Callable[[ModelCall], ModelResponse],
    ) -> ModelResponse:
        """Intercept model execution via handler callback."""
        ...


class _CallableReturningToolResponse(Protocol):
    """Callable for tool call interception with handler callback.

    Receives handler callback to execute tool and returns final ToolMessage or Command.
    """

    def __call__(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
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
        can_jump_to: Optional list of valid jump destinations. Can be:
            - "tools": Jump to the tools node
            - "model": Jump back to the model node
            - "end": Jump to the end of the graph

    Returns:
        Decorator function that marks the method with configuration metadata.

    Examples:
        Using decorator on a class method:
        ```python
        class MyMiddleware(AgentMiddleware):
            @hook_config(can_jump_to=["end", "model"])
            def before_model(self, state: AgentState) -> dict[str, Any] | None:
                if some_condition(state):
                    return {"jump_to": "end"}
                return None
        ```

        Alternative: Use the `can_jump_to` parameter in `before_model`/`after_model` decorators:
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
    """Decorator used to dynamically create a middleware with the before_model hook.

    Args:
        func: The function to be decorated. Must accept:
            `state: StateT, runtime: Runtime[ContextT]` - State and runtime context
        state_schema: Optional custom state schema type. If not provided, uses the default
            AgentState schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.
            Valid values are: "tools", "model", "end"
        name: Optional name for the generated middleware class. If not provided,
            uses the decorated function's name.

    Returns:
        Either an AgentMiddleware instance (if func is provided directly) or a decorator function
        that can be applied to a function it is wrapping.

    The decorated function should return:
        - `dict[str, Any]` - State updates to merge into the agent state
        - `Command` - A command to control flow (e.g., jump to different node)
        - `None` - No state updates or flow control

    Examples:
        Basic usage:
        ```python
        @before_model
        def log_before_model(state: AgentState, runtime: Runtime) -> None:
            print(f"About to call model with {len(state['messages'])} messages")
        ```

        With conditional jumping:
        ```python
        @before_model(can_jump_to=["end"])
        def conditional_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            if some_condition(state):
                return {"jump_to": "end"}
            return None
        ```

        With custom state schema:
        ```python
        @before_model(state_schema=MyCustomState)
        def custom_before_model(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
            return {"custom_field": "updated_value"}
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
                self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command | None:
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
            self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command | None:
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
    """Decorator used to dynamically create a middleware with the after_model hook.

    Args:
        func: The function to be decorated. Must accept:
            `state: StateT, runtime: Runtime[ContextT]` - State and runtime context
        state_schema: Optional custom state schema type. If not provided, uses the default
            AgentState schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.
            Valid values are: "tools", "model", "end"
        name: Optional name for the generated middleware class. If not provided,
            uses the decorated function's name.

    Returns:
        Either an AgentMiddleware instance (if func is provided) or a decorator function
        that can be applied to a function.

    The decorated function should return:
        - `dict[str, Any]` - State updates to merge into the agent state
        - `Command` - A command to control flow (e.g., jump to different node)
        - `None` - No state updates or flow control

    Examples:
        Basic usage for logging model responses:
        ```python
        @after_model
        def log_latest_message(state: AgentState, runtime: Runtime) -> None:
            print(state["messages"][-1].content)
        ```

        With custom state schema:
        ```python
        @after_model(state_schema=MyCustomState, name="MyAfterModelMiddleware")
        def custom_after_model(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
            return {"custom_field": "updated_after_model"}
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
                self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command | None:
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
            self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command | None:
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
    """Decorator used to dynamically create a middleware with the before_agent hook.

    Args:
        func: The function to be decorated. Must accept:
            `state: StateT, runtime: Runtime[ContextT]` - State and runtime context
        state_schema: Optional custom state schema type. If not provided, uses the default
            AgentState schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.
            Valid values are: "tools", "model", "end"
        name: Optional name for the generated middleware class. If not provided,
            uses the decorated function's name.

    Returns:
        Either an AgentMiddleware instance (if func is provided directly) or a decorator function
        that can be applied to a function it is wrapping.

    The decorated function should return:
        - `dict[str, Any]` - State updates to merge into the agent state
        - `Command` - A command to control flow (e.g., jump to different node)
        - `None` - No state updates or flow control

    Examples:
        Basic usage:
        ```python
        @before_agent
        def log_before_agent(state: AgentState, runtime: Runtime) -> None:
            print(f"Starting agent with {len(state['messages'])} messages")
        ```

        With conditional jumping:
        ```python
        @before_agent(can_jump_to=["end"])
        def conditional_before_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            if some_condition(state):
                return {"jump_to": "end"}
            return None
        ```

        With custom state schema:
        ```python
        @before_agent(state_schema=MyCustomState)
        def custom_before_agent(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
            return {"custom_field": "initialized_value"}
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
                self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command | None:
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
            self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command | None:
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
    """Decorator used to dynamically create a middleware with the after_agent hook.

    Args:
        func: The function to be decorated. Must accept:
            `state: StateT, runtime: Runtime[ContextT]` - State and runtime context
        state_schema: Optional custom state schema type. If not provided, uses the default
            AgentState schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.
            Valid values are: "tools", "model", "end"
        name: Optional name for the generated middleware class. If not provided,
            uses the decorated function's name.

    Returns:
        Either an AgentMiddleware instance (if func is provided) or a decorator function
        that can be applied to a function.

    The decorated function should return:
        - `dict[str, Any]` - State updates to merge into the agent state
        - `Command` - A command to control flow (e.g., jump to different node)
        - `None` - No state updates or flow control

    Examples:
        Basic usage for logging agent completion:
        ```python
        @after_agent
        def log_completion(state: AgentState, runtime: Runtime) -> None:
            print(f"Agent completed with {len(state['messages'])} messages")
        ```

        With custom state schema:
        ```python
        @after_agent(state_schema=MyCustomState, name="MyAfterAgentMiddleware")
        def custom_after_agent(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
            return {"custom_field": "finalized_value"}
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
                self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command | None:
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
            self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command | None:
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
    func: _CallableReturningPromptString[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def dynamic_prompt(
    func: None = None,
) -> Callable[
    [_CallableReturningPromptString[StateT, ContextT]],
    AgentMiddleware[StateT, ContextT],
]: ...


def dynamic_prompt(
    func: _CallableReturningPromptString[StateT, ContextT] | None = None,
) -> (
    Callable[
        [_CallableReturningPromptString[StateT, ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically generate system prompts for the model.

    This is a convenience decorator that creates middleware using `wrap_model_call`
    specifically for dynamic prompt generation. The decorated function should return
    a string that will be set as the system prompt for the model request.

    Args:
        func: The function to be decorated. Must accept:
            `request: ModelRequest` - Model request (contains state and runtime)

    Returns:
        Either an AgentMiddleware instance (if func is provided) or a decorator function
        that can be applied to a function.

    The decorated function should return:
        - `str` - The system prompt to use for the model request

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
        func: _CallableReturningPromptString[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        if is_async:

            async def async_wrapped(
                self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
                request: ModelRequest,
                handler: Callable[[ModelCall], Awaitable[ModelResponse]],
            ) -> ModelResponse:
                prompt = await func(request)  # type: ignore[misc]
                request.model_call.system_prompt = prompt
                return await handler(request.model_call)

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
            self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
            request: ModelRequest,
            handler: Callable[[ModelCall], ModelResponse],
        ) -> ModelResponse:
            prompt = cast("str", func(request))
            request.model_call.system_prompt = prompt
            return handler(request.model_call)

        middleware_name = cast("str", getattr(func, "__name__", "DynamicPromptMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": AgentState,
                "tools": [],
                "wrap_model_call": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def wrap_model_call(
    func: _CallableReturningModelResponse[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def wrap_model_call(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableReturningModelResponse[StateT, ContextT]],
    AgentMiddleware[StateT, ContextT],
]: ...


def wrap_model_call(
    func: _CallableReturningModelResponse[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableReturningModelResponse[StateT, ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
):
    """Create middleware with wrap_model_call hook from a function.

    Converts a function with handler callback into middleware that can intercept
    model calls, implement retry logic, handle errors, and rewrite responses.

    Args:
        func: Function accepting (request, handler) that calls handler(request)
            to execute the model and returns final AIMessage. Request contains state and runtime.
        state_schema: Custom state schema. Defaults to AgentState.
        tools: Additional tools to register with this middleware.
        name: Middleware class name. Defaults to function name.

    Returns:
        AgentMiddleware instance if func provided, otherwise a decorator.

    Examples:
        Basic retry logic:
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

        Model fallback:
        ```python
        @wrap_model_call
        def fallback_model(request, handler):
            # Try primary model
            try:
                return handler(request)
            except Exception:
                pass

            # Try fallback model
            request.model = fallback_model_instance
            return handler(request)
        ```

        Rewrite response content:
        ```python
        @wrap_model_call
        def uppercase_responses(request, handler):
            result = handler(request)
            return AIMessage(content=result.content.upper())
        ```
    """

    def decorator(
        func: _CallableReturningModelResponse[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        if is_async:

            async def async_wrapped(
                self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
                request: ModelRequest,
                handler: Callable[[ModelCall], Awaitable[ModelResponse]],
            ) -> ModelResponse:
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
            self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
            request: ModelRequest,
            handler: Callable[[ModelCall], ModelResponse],
        ) -> ModelResponse:
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
    """Create middleware with wrap_tool_call hook from a function.

    Converts a function with handler callback into middleware that can intercept
    tool calls, implement retry logic, monitor execution, and modify responses.

    Args:
        func: Function accepting (request, handler) that calls
            handler(request) to execute the tool and returns final ToolMessage or Command.
        tools: Additional tools to register with this middleware.
        name: Middleware class name. Defaults to function name.

    Returns:
        AgentMiddleware instance if func provided, otherwise a decorator.

    Examples:
        Basic passthrough:
        ```python
        @wrap_tool_call
        def passthrough(request, handler):
            return handler(request)
        ```

        Retry logic:
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

        Modify request:
        ```python
        @wrap_tool_call
        def modify_args(request, handler):
            request.tool_call["args"]["value"] *= 2
            return handler(request)
        ```

        Short-circuit with cached result:
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
        def wrapped(
            self: AgentMiddleware,  # noqa: ARG001
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command],
        ) -> ToolMessage | Command:
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
