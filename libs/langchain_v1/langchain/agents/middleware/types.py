"""Types for middleware and agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeGuard,
    cast,
    overload,
)

# needed as top level import for pydantic schema generation on AgentState
from langchain_core.messages import AnyMessage  # noqa: TC002
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.typing import ContextT
from typing_extensions import NotRequired, Required, TypedDict, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime
    from langgraph.types import Command

    from langchain.agents.structured_output import ResponseFormat

__all__ = [
    "AgentMiddleware",
    "AgentState",
    "ContextT",
    "ModelRequest",
    "OmitFromSchema",
    "PublicAgentState",
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
    tools: list[str]
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
    response: NotRequired[ResponseT]


class PublicAgentState(TypedDict, Generic[ResponseT]):
    """Public state schema for the agent.

    Just used for typing purposes.
    """

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    response: NotRequired[ResponseT]


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

    before_model_jump_to: ClassVar[list[JumpTo]] = []
    """Valid jump destinations for before_model hook. Used to establish conditional edges."""

    after_model_jump_to: ClassVar[list[JumpTo]] = []
    """Valid jump destinations for after_model hook. Used to establish conditional edges."""

    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the model is called."""

    def modify_model_request(
        self,
        request: ModelRequest,
        state: StateT,  # noqa: ARG002
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> ModelRequest:
        """Logic to modify request kwargs before the model is called."""
        return request

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the model is called."""


class _CallableWithState(Protocol[StateT_contra]):
    """Callable with AgentState as argument."""

    def __call__(self, state: StateT_contra) -> dict[str, Any] | Command | None:
        """Perform some logic with the state."""
        ...


class _CallableWithStateAndRuntime(Protocol[StateT_contra, ContextT]):
    """Callable with AgentState and Runtime as arguments."""

    def __call__(
        self, state: StateT_contra, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | Command | None:
        """Perform some logic with the state and runtime."""
        ...


class _CallableWithModelRequestAndState(Protocol[StateT_contra]):
    """Callable with ModelRequest and AgentState as arguments."""

    def __call__(self, request: ModelRequest, state: StateT_contra) -> ModelRequest:
        """Perform some logic with the model request and state."""
        ...


class _CallableWithModelRequestAndStateAndRuntime(Protocol[StateT_contra, ContextT]):
    """Callable with ModelRequest, AgentState, and Runtime as arguments."""

    def __call__(
        self, request: ModelRequest, state: StateT_contra, runtime: Runtime[ContextT]
    ) -> ModelRequest:
        """Perform some logic with the model request, state, and runtime."""
        ...


_NodeSignature: TypeAlias = (
    _CallableWithState[StateT] | _CallableWithStateAndRuntime[StateT, ContextT]
)
_ModelRequestSignature: TypeAlias = (
    _CallableWithModelRequestAndState[StateT]
    | _CallableWithModelRequestAndStateAndRuntime[StateT, ContextT]
)


def is_callable_with_runtime(
    func: _NodeSignature[StateT, ContextT],
) -> TypeGuard[_CallableWithStateAndRuntime[StateT, ContextT]]:
    return "runtime" in signature(func).parameters


def is_callable_with_runtime_and_request(
    func: _ModelRequestSignature[StateT, ContextT],
) -> TypeGuard[_CallableWithModelRequestAndStateAndRuntime[StateT, ContextT]]:
    return "runtime" in signature(func).parameters


@overload
def before_model(
    func: _NodeSignature[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def before_model(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[[_NodeSignature[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]: ...


def before_model(
    func: _NodeSignature[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_NodeSignature[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the before_model hook.

    Args:
        func: The function to be decorated. Can accept either:
            - `state: StateT` - Just the agent state
            - `state: StateT, runtime: Runtime[ContextT]` - State and runtime context
        state_schema: Optional custom state schema type. If not provided, uses the default
            AgentState schema.
        tools: Optional list of additional tools to register with this middleware.
        jump_to: Optional list of valid jump destinations for conditional edges.
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
        Basic usage with state only:
        ```python
        @before_model
        def log_before_model(state: AgentState) -> None:
            print(f"About to call model with {len(state['messages'])} messages")
        ```

        Advanced usage with runtime and conditional jumping:
        ```python
        @before_model(jump_to=["end"])
        def conditional_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            if some_condition(state):
                return {"jump_to": "end"}
            return None
        ```

        With custom state schema:
        ```python
        @before_model(
            state_schema=MyCustomState,
        )
        def custom_before_model(state: MyCustomState) -> dict[str, Any]:
            return {"custom_field": "updated_value"}
        ```
    """

    def decorator(func: _NodeSignature[StateT, ContextT]) -> AgentMiddleware[StateT, ContextT]:
        if is_callable_with_runtime(func):

            def wrapped_with_runtime(
                self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command | None:
                return func(state, runtime)

            wrapped = wrapped_with_runtime
        else:

            def wrapped_without_runtime(
                self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
                state: StateT,
            ) -> dict[str, Any] | Command | None:
                return func(state)  # type: ignore[call-arg]

            wrapped = wrapped_without_runtime  # type: ignore[assignment]

        # Use function name as default if no name provided
        middleware_name = name or cast("str", getattr(func, "__name__", "BeforeModelMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "before_model_jump_to": jump_to or [],
                "before_model": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def modify_model_request(
    func: _ModelRequestSignature[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def modify_model_request(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> Callable[[_ModelRequestSignature[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]: ...


def modify_model_request(
    func: _ModelRequestSignature[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> (
    Callable[[_ModelRequestSignature[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    r"""Decorator used to dynamically create a middleware with the modify_model_request hook.

    Args:
        func: The function to be decorated. Can accept either:
            - `request: ModelRequest, state: StateT` - Model request and agent state
            - `request: ModelRequest, state: StateT, runtime: Runtime[ContextT]` -
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
        def add_context_to_prompt(request: ModelRequest, state: AgentState) -> ModelRequest:
            if request.system_prompt:
                request.system_prompt += "\n\nAdditional context: ..."
            else:
                request.system_prompt = "Additional context: ..."
            return request
        ```

        Advanced usage with runtime and custom model settings:
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
        func: _ModelRequestSignature[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        if is_callable_with_runtime_and_request(func):

            def wrapped_with_runtime(
                self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
                request: ModelRequest,
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> ModelRequest:
                return func(request, state, runtime)

            wrapped = wrapped_with_runtime
        else:

            def wrapped_without_runtime(
                self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
                request: ModelRequest,
                state: StateT,
            ) -> ModelRequest:
                return func(request, state)  # type: ignore[call-arg]

            wrapped = wrapped_without_runtime  # type: ignore[assignment]

        # Use function name as default if no name provided
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
    func: _NodeSignature[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def after_model(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[[_NodeSignature[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]: ...


def after_model(
    func: _NodeSignature[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_NodeSignature[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the after_model hook.

    Args:
        func: The function to be decorated. Can accept either:
            - `state: StateT` - Just the agent state (includes model response)
            - `state: StateT, runtime: Runtime[ContextT]` - State and runtime context
        state_schema: Optional custom state schema type. If not provided, uses the default
            AgentState schema.
        tools: Optional list of additional tools to register with this middleware.
        jump_to: Optional list of valid jump destinations for conditional edges.
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
        def log_latest_message(state: AgentState) -> None:
            print(state["messages"][-1].content)
        ```

        With custom state schema:
        ```python
        @after_model(state_schema=MyCustomState, name="MyAfterModelMiddleware")
        def custom_after_model(state: MyCustomState) -> dict[str, Any]:
            return {"custom_field": "updated_after_model"}
        ```
    """

    def decorator(func: _NodeSignature[StateT, ContextT]) -> AgentMiddleware[StateT, ContextT]:
        if is_callable_with_runtime(func):

            def wrapped_with_runtime(
                self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command | None:
                return func(state, runtime)

            wrapped = wrapped_with_runtime
        else:

            def wrapped_without_runtime(
                self: AgentMiddleware[StateT, ContextT],  # noqa: ARG001
                state: StateT,
            ) -> dict[str, Any] | Command | None:
                return func(state)  # type: ignore[call-arg]

            wrapped = wrapped_without_runtime  # type: ignore[assignment]

        # Use function name as default if no name provided
        middleware_name = name or cast("str", getattr(func, "__name__", "AfterModelMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "after_model_jump_to": jump_to or [],
                "after_model": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator
