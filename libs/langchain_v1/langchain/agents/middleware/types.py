"""Types for middleware and agents."""

from __future__ import annotations

from collections.abc import Callable, Generator
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

from langchain_core.runnables import run_in_executor

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from langchain.tools.tool_node import ToolCallRequest

# needed as top level import for pydantic schema generation on AgentState
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage  # noqa: TC002
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
    "ModelRequest",
    "OmitFromSchema",
    "PublicAgentState",
    "after_agent",
    "after_model",
    "before_agent",
    "before_model",
    "dynamic_prompt",
    "hook_config",
    "modify_model_request",
    "on_model_call",
    "on_tool_call",
]

JumpTo = Literal["tools", "model", "end"]
"""Destination to jump to when a middleware node returns."""

ResponseT = TypeVar("ResponseT")


@dataclass
class ModelRequest:
    """Model request information for the agent."""

    model: BaseChatModel
    system_prompt: str | None
    messages: list[AnyMessage]  # excluding system prompt
    tool_choice: Any | None
    tools: list[BaseTool | dict]
    response_format: ResponseFormat | None
    model_settings: dict[str, Any] = field(default_factory=dict)


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

    def modify_model_request(
        self,
        request: ModelRequest,
        state: StateT,  # noqa: ARG002
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> ModelRequest:
        """Logic to modify request kwargs before the model is called."""
        return request

    async def amodify_model_request(
        self,
        request: ModelRequest,
        state: StateT,
        runtime: Runtime[ContextT],
    ) -> ModelRequest:
        """Async logic to modify request kwargs before the model is called."""
        return await run_in_executor(None, self.modify_model_request, request, state, runtime)

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the model is called."""

    async def aafter_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run after the model is called."""

    def on_model_call(
        self,
        request: ModelRequest,
        state: StateT,
        runtime: Runtime[ContextT],
    ) -> Generator[ModelRequest | AIMessage, AIMessage, None]:
        """Intercept and control model execution via generator protocol.

        Protocol:
        1. Yield ModelRequest to execute the model.
        2. Receive AIMessage via .send() on success, or exception via .throw() on error.
        3. Optionally yield again to retry.
        4. Generator ends naturally - consumer uses last successful AIMessage.

        Middleware can implement retry logic, error handling, response rewriting,
        and request modification using standard try/except. Multiple middleware
        compose with first in list as outermost layer.

        Args:
            request: Initial model request to execute.
            state: Current agent state.
            runtime: LangGraph runtime context.

        Yields:
            ModelRequest to execute.

        Receives:
            AIMessage via .send() on success.
            Exception via .throw() on error.

        Examples:
            Retry on error:
            ```python
            def on_model_call(self, request, state, runtime):
                for attempt in range(3):
                    try:
                        yield request
                        break  # Success
                    except Exception:
                        if attempt == 2:
                            raise
            ```

            Rewrite response:
            ```python
            def on_model_call(self, request, state, runtime):
                result = yield request
                modified = AIMessage(content=f"[{result.content}]")
                yield modified
            ```

            Error to fallback:
            ```python
            def on_model_call(self, request, state, runtime):
                try:
                    yield request
                except Exception:
                    fallback = AIMessage(content="Service unavailable")
                    yield fallback

            Cache/short-circuit:
            ```python
            def on_model_call(self, request, state, runtime):
                if cached := get_cache(request):
                    yield cached  # Short-circuit with cached result
                else:
                    result = yield request
                    save_cache(request, result)
            ```
        """
        raise NotImplementedError

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the agent execution completes."""

    async def aafter_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run after the agent execution completes."""

    def on_tool_call(
        self,
        request: ToolCallRequest,
        state: StateT,  # noqa: ARG002
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Intercept tool execution for retries, monitoring, or modification.

        Multiple middleware compose automatically (first defined = outermost).
        Exceptions propagate unless handle_tool_errors is configured on ToolNode.

        Args:
            request: Tool call request with call dict and BaseTool instance.
            state: Current agent state.
            runtime: LangGraph runtime.

        Yields:
            ToolCallRequest (execute tool), ToolMessage (cached result),
            or Command (control flow).

        Receives:
            ToolMessage or Command via .send() after execution.

        Example:
            Modify request:

            def on_tool_call(self, request, state, runtime):
                request.tool_call["args"]["value"] *= 2
                yield request

            Retry on error:

            def on_tool_call(self, request, state, runtime):
                for attempt in range(3):
                    response = yield request
                    if valid(response) or attempt == 2:
                        return
        """
        yield request


class _CallableWithStateAndRuntime(Protocol[StateT_contra, ContextT]):
    """Callable with AgentState and Runtime as arguments."""

    def __call__(
        self, state: StateT_contra, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | Command | None | Awaitable[dict[str, Any] | Command | None]:
        """Perform some logic with the state and runtime."""
        ...


class _CallableWithModelRequestAndStateAndRuntime(Protocol[StateT_contra, ContextT]):
    """Callable with ModelRequest, AgentState, and Runtime as arguments."""

    def __call__(
        self, request: ModelRequest, state: StateT_contra, runtime: Runtime[ContextT]
    ) -> ModelRequest | Awaitable[ModelRequest]:
        """Perform some logic with the model request, state, and runtime."""
        ...


class _CallableReturningPromptString(Protocol[StateT_contra, ContextT]):
    """Callable that returns a prompt string given ModelRequest, AgentState, and Runtime."""

    def __call__(
        self, request: ModelRequest, state: StateT_contra, runtime: Runtime[ContextT]
    ) -> str | Awaitable[str]:
        """Generate a system prompt string based on the request, state, and runtime."""
        ...


class _CallableReturningModelResponseGenerator(Protocol[StateT_contra, ContextT]):
    """Callable returning generator for model call interception.

    Returns sync generator that works with both sync and async model execution.
    Generator receives AIMessage via .send() or exception via .throw().
    """

    def __call__(
        self, request: ModelRequest, state: StateT_contra, runtime: Runtime[ContextT]
    ) -> Generator[ModelRequest, AIMessage, AIMessage]:
        """Return generator to intercept model execution."""
        ...


class _CallableReturningToolResponseGenerator(Protocol[StateT_contra, ContextT]):
    """Callable returning generator for tool call interception.

    Returns sync generator that works with both sync and async tool execution.
    Generator receives ToolMessage or Command via .send().
    """

    def __call__(
        self, request: ToolCallRequest, state: StateT_contra, runtime: Runtime[ContextT]
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Return generator to intercept tool execution."""
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
def modify_model_request(
    func: _CallableWithModelRequestAndStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def modify_model_request(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithModelRequestAndStateAndRuntime[StateT, ContextT]],
    AgentMiddleware[StateT, ContextT],
]: ...


def modify_model_request(
    func: _CallableWithModelRequestAndStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableWithModelRequestAndStateAndRuntime[StateT, ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
):
    r"""Decorator used to dynamically create a middleware with the modify_model_request hook.

    Args:
        func: The function to be decorated. Must accept:
            `request: ModelRequest, state: StateT, runtime: Runtime[ContextT]` -
            Model request, state, and runtime context
        state_schema: Optional custom state schema type. If not provided, uses the default
            AgentState schema.
        tools: Optional list of additional tools to register with this middleware.
        name: Optional name for the generated middleware class. If not provided,
            uses the decorated function's name.

    Returns:
        Either an AgentMiddleware instance (if func is provided) or a decorator function
        that can be applied to a function.

    The decorated function should return:
        - `ModelRequest` - The modified model request to be sent to the language model

    Examples:
        Basic usage to modify system prompt:
        ```python
        @modify_model_request
        def add_context_to_prompt(
            request: ModelRequest, state: AgentState, runtime: Runtime
        ) -> ModelRequest:
            if request.system_prompt:
                request.system_prompt += "\n\nAdditional context: ..."
            else:
                request.system_prompt = "Additional context: ..."
            return request
        ```

        Usage with runtime and custom model settings:
        ```python
        @modify_model_request
        def dynamic_model_settings(
            request: ModelRequest, state: AgentState, runtime: Runtime
        ) -> ModelRequest:
            # Use a different model based on user subscription tier
            if runtime.context.get("subscription_tier") == "premium":
                request.model = "gpt-4o"
            else:
                request.model = "gpt-4o-mini"

            return request
        ```
    """

    def decorator(
        func: _CallableWithModelRequestAndStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        if is_async:

            async def async_wrapped(
                self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
                request: ModelRequest,
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> ModelRequest:
                return await func(request, state, runtime)  # type: ignore[misc]

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "ModifyModelRequestMiddleware")
            )

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "amodify_model_request": async_wrapped,
                },
            )()

        def wrapped(
            self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
            request: ModelRequest,
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> ModelRequest:
            return func(request, state, runtime)  # type: ignore[return-value]

        middleware_name = name or cast(
            "str", getattr(func, "__name__", "ModifyModelRequestMiddleware")
        )

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "modify_model_request": wrapped,
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

    This is a convenience decorator that creates middleware using `modify_model_request`
    specifically for dynamic prompt generation. The decorated function should return
    a string that will be set as the system prompt for the model request.

    Args:
        func: The function to be decorated. Must accept:
            `request: ModelRequest, state: StateT, runtime: Runtime[ContextT]` -
            Model request, state, and runtime context

    Returns:
        Either an AgentMiddleware instance (if func is provided) or a decorator function
        that can be applied to a function.

    The decorated function should return:
        - `str` - The system prompt to use for the model request

    Examples:
        Basic usage with dynamic content:
        ```python
        @dynamic_prompt
        def my_prompt(request: ModelRequest, state: AgentState, runtime: Runtime) -> str:
            user_name = runtime.context.get("user_name", "User")
            return f"You are a helpful assistant helping {user_name}."
        ```

        Using state to customize the prompt:
        ```python
        @dynamic_prompt
        def context_aware_prompt(request: ModelRequest, state: AgentState, runtime: Runtime) -> str:
            msg_count = len(state["messages"])
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
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> ModelRequest:
                prompt = await func(request, state, runtime)  # type: ignore[misc]
                request.system_prompt = prompt
                return request

            middleware_name = cast("str", getattr(func, "__name__", "DynamicPromptMiddleware"))

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": AgentState,
                    "tools": [],
                    "amodify_model_request": async_wrapped,
                },
            )()

        def wrapped(
            self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
            request: ModelRequest,
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> ModelRequest:
            prompt = cast("str", func(request, state, runtime))
            request.system_prompt = prompt
            return request

        middleware_name = cast("str", getattr(func, "__name__", "DynamicPromptMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": AgentState,
                "tools": [],
                "modify_model_request": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def on_model_call(
    func: _CallableReturningModelResponseGenerator[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def on_model_call(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableReturningModelResponseGenerator[StateT, ContextT]],
    AgentMiddleware[StateT, ContextT],
]: ...


def on_model_call(
    func: _CallableReturningModelResponseGenerator[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableReturningModelResponseGenerator[StateT, ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
):
    """Create middleware with on_model_call hook from a generator function.

    Converts a generator function into middleware that can intercept model calls,
    implement retry logic, handle errors, and rewrite responses using standard
    Python exception handling.

    Args:
        func: Generator function accepting (request, state, runtime) that yields
            ModelRequest, receives AIMessage via .send() on success or exception
            via .throw() on error, and returns final AIMessage.
        state_schema: Custom state schema. Defaults to AgentState.
        tools: Additional tools to register with this middleware.
        name: Middleware class name. Defaults to function name.

    Returns:
        AgentMiddleware instance if func provided, otherwise a decorator.

    Examples:
        Basic retry logic:
        ```python
        @on_model_call
        def retry_on_error(
            request: ModelRequest, state: AgentState, runtime: Runtime
        ) -> Generator[ModelRequest, AIMessage, AIMessage]:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    yield request
                    break  # Success
                except Exception:
                    if attempt == max_retries - 1:
                        raise
        ```

        Model fallback:
        ```python
        @on_model_call
        def fallback_model(
            request: ModelRequest, state: AgentState, runtime: Runtime
        ) -> Generator[ModelRequest, AIMessage, AIMessage]:
            # Try primary model
            try:
                yield request
                return  # Success
            except Exception:
                pass

            # Try fallback model
            request.model = fallback_model_instance
            yield request
        ```

        Rewrite response content:
        ```python
        @on_model_call
        def uppercase_responses(
            request: ModelRequest, state: AgentState, runtime: Runtime
        ) -> Generator[ModelRequest, AIMessage, AIMessage]:
            result = yield request
            modified = AIMessage(content=result.content.upper())
            yield modified
        ```
    """

    def decorator(
        func: _CallableReturningModelResponseGenerator[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        def wrapped(
            self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
            request: ModelRequest,
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> Generator[ModelRequest, AIMessage, AIMessage]:
            return func(request, state, runtime)

        middleware_name = name or cast("str", getattr(func, "__name__", "OnModelCallMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "on_model_call": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def on_tool_call(
    func: _CallableReturningToolResponseGenerator[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def on_tool_call(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableReturningToolResponseGenerator[StateT, ContextT]],
    AgentMiddleware[StateT, ContextT],
]: ...


def on_tool_call(
    func: _CallableReturningToolResponseGenerator[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableReturningToolResponseGenerator[StateT, ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
):
    """Create middleware with on_tool_call hook from a generator function.

    Converts a generator function into middleware that can intercept tool calls,
    implement retry logic, modify requests, cache results, or handle errors using
    standard Python exception handling.

    Args:
        func: Generator function accepting (request, state, runtime) that yields
            ToolCallRequest, ToolMessage, or Command, receives ToolMessage or Command
            via .send() on completion, and optionally handles exceptions via .throw().
        state_schema: Custom state schema. Defaults to AgentState.
        tools: Additional tools to register with this middleware.
        name: Middleware class name. Defaults to function name.

    Returns:
        AgentMiddleware instance if func provided, otherwise a decorator.

    Examples:
        Basic retry logic:
        ```python
        @on_tool_call
        def retry_on_error(
            request: ToolCallRequest, state: AgentState, runtime: Runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            max_retries = 3
            for attempt in range(max_retries):
                response = yield request
                if isinstance(response, ToolMessage) and not response.status == "error":
                    return
                if attempt == max_retries - 1:
                    return
        ```

        Modify tool arguments:
        ```python
        @on_tool_call
        def scale_numeric_args(
            request: ToolCallRequest, state: AgentState, runtime: Runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            # Double numeric arguments
            for key, value in request.tool_call["args"].items():
                if isinstance(value, (int, float)):
                    request.tool_call["args"][key] = value * 2
            yield request
        ```

        Cache tool results:
        ```python
        @on_tool_call
        def cache_results(
            request: ToolCallRequest, state: AgentState, runtime: Runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            cache_key = f"{request.tool_call['name']}:{request.tool_call['args']}"
            if cached := get_cache(cache_key):
                yield cached  # Short-circuit with cached result
            else:
                result = yield request
                save_cache(cache_key, result)
        ```

        Error handling with fallback:
        ```python
        @on_tool_call
        def handle_errors(
            request: ToolCallRequest, state: AgentState, runtime: Runtime
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            try:
                yield request
            except Exception as e:
                # Return error message instead of propagating
                yield ToolMessage(
                    content=f"Tool execution failed: {e}",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
        ```
    """

    def decorator(
        func: _CallableReturningToolResponseGenerator[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        def wrapped(
            self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
            request: ToolCallRequest,
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
            return func(request, state, runtime)

        middleware_name = name or cast("str", getattr(func, "__name__", "OnToolCallMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "on_tool_call": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator
