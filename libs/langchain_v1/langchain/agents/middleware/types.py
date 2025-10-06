"""Types for middleware and agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
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

# needed as top level import for pydantic schema generation on AgentState
from langchain_core.messages import AnyMessage  # noqa: TC002
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.graph.message import add_messages
from langgraph.typing import ContextT
from typing_extensions import NotRequired, Required, TypedDict, TypeVar

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
    "HookImplementation",
    "MiddlewareHookInfo",
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
]

JumpTo = Literal["tools", "model", "end"]
"""Destination to jump to when a middleware node returns."""

ResponseT = TypeVar("ResponseT")


class HookImplementation(str, Enum):
    """Tracks which implementation variants exist for a middleware hook."""

    NONE = "none"
    SYNC_ONLY = "sync"
    ASYNC_ONLY = "async"
    BOTH = "both"


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


@dataclass
class MiddlewareHookInfo:
    """Information about a specific middleware hook implementation.

    This class encapsulates metadata about how a middleware implements a particular hook,
    including the actual hook functions and jump configuration.
    """

    middleware_name: str
    """The name of the middleware that implements this hook."""

    hook_name: str
    """The name of the hook (e.g., 'before_model', 'after_agent')."""

    sync_fn: Callable[..., Any] | None
    """The synchronous hook function, or None if not implemented."""

    async_fn: Callable[..., Any] | None
    """The asynchronous hook function, or None if not implemented."""

    can_jump_to: list[JumpTo]
    """Valid jump destinations for this hook."""

    @property
    def node_name(self) -> str:
        """The graph node name for this hook."""
        return f"{self.middleware_name}.{self.hook_name}"

    @property
    def has_sync(self) -> bool:
        """Whether this hook has a sync implementation."""
        return self.sync_fn is not None

    @property
    def has_async(self) -> bool:
        """Whether this hook has an async implementation."""
        return self.async_fn is not None

    @property
    def implementation(self) -> HookImplementation:
        """Which variants (sync/async/both) are implemented."""
        if self.has_sync and self.has_async:
            return HookImplementation.BOTH
        if self.has_sync:
            return HookImplementation.SYNC_ONLY
        if self.has_async:
            return HookImplementation.ASYNC_ONLY
        return HookImplementation.NONE


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

    def retry_model_request(
        self,
        error: Exception,  # noqa: ARG002
        request: ModelRequest,  # noqa: ARG002
        state: StateT,  # noqa: ARG002
        runtime: Runtime[ContextT],  # noqa: ARG002
        attempt: int,  # noqa: ARG002
    ) -> ModelRequest | None:
        """Logic to handle model invocation errors and optionally retry.

        Args:
            error: The exception that occurred during model invocation.
            request: The original model request that failed.
            state: The current agent state.
            runtime: The langgraph runtime.
            attempt: The current attempt number (1-indexed).

        Returns:
            ModelRequest: Modified request to retry with.
            None: Propagate the error (re-raise).
        """
        return None

    async def aretry_model_request(
        self,
        error: Exception,
        request: ModelRequest,
        state: StateT,
        runtime: Runtime[ContextT],
        attempt: int,
    ) -> ModelRequest | None:
        """Async logic to handle model invocation errors and optionally retry.

        Args:
            error: The exception that occurred during model invocation.
            request: The original model request that failed.
            state: The current agent state.
            runtime: The langgraph runtime.
            attempt: The current attempt number (1-indexed).

        Returns:
            ModelRequest: Modified request to retry with.
            None: Propagate the error (re-raise).
        """
        return await run_in_executor(
            None, self.retry_model_request, error, request, state, runtime, attempt
        )

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the agent execution completes."""

    async def aafter_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run after the agent execution completes."""

    def hook_info(self, hook_name: str) -> MiddlewareHookInfo | None:
        """Get information about this middleware's implementation of a specific hook.

        Args:
            hook_name: The name of the hook to inspect (e.g., 'before_model', 'after_agent').

        Returns:
            MiddlewareHookInfo if the hook is implemented, None otherwise.

        Example:
            >>> middleware = MyMiddleware()
            >>> info = middleware.hook_info("before_model")
            >>> if info:
            ...     print(f"Has sync: {info.has_sync}, Has async: {info.has_async}")
        """
        base_class = AgentMiddleware
        middleware_class = self.__class__

        # Check sync and async variants
        sync_name = hook_name
        async_name = f"a{hook_name}"

        base_sync_method = getattr(base_class, sync_name, None)
        base_async_method = getattr(base_class, async_name, None)

        middleware_sync_method = getattr(middleware_class, sync_name, None)
        middleware_async_method = getattr(middleware_class, async_name, None)

        has_custom_sync = middleware_sync_method is not base_sync_method
        has_custom_async = middleware_async_method is not base_async_method

        if not has_custom_sync and not has_custom_async:
            return None

        # Get the actual bound methods - only include customized implementations
        sync_fn = getattr(self, sync_name) if has_custom_sync else None
        async_fn = getattr(self, async_name) if has_custom_async else None

        # Get can_jump_to from either sync or async variant
        can_jump_to: list[JumpTo] = []
        if has_custom_sync:
            can_jump_to = getattr(middleware_sync_method, "__can_jump_to__", [])
        elif has_custom_async:
            can_jump_to = getattr(middleware_async_method, "__can_jump_to__", [])

        return MiddlewareHookInfo(
            middleware_name=self.name,
            hook_name=hook_name,
            sync_fn=sync_fn,
            async_fn=async_fn,
            can_jump_to=can_jump_to,
        )

    def all_hook_info(self) -> dict[str, MiddlewareHookInfo]:
        """Get information about all hooks implemented by this middleware.

        Returns:
            Dictionary mapping hook names to their MiddlewareHookInfo.

        Example:
            >>> middleware = MyMiddleware()
            >>> for hook_name, info in middleware.all_hook_info().items():
            ...     print(f"{hook_name}: sync={info.has_sync}, async={info.has_async}")
        """
        hook_names = [
            "before_agent",
            "before_model",
            "modify_model_request",
            "after_model",
            "after_agent",
            "retry_model_request",
        ]
        return {name: info for name in hook_names if (info := self.hook_info(name)) is not None}

    @property
    def implemented_hooks(self) -> list[str]:
        """List of hook names this middleware implements.

        Returns:
            List of hook names that are overridden from the base class.

        Example:
            >>> middleware = MyMiddleware()
            >>> print(middleware.implemented_hooks)
            ['before_model', 'after_model']
        """
        return list(self.all_hook_info().keys())


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
        that can be applied to a function its wrapping.

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
        that can be applied to a function its wrapping.

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
