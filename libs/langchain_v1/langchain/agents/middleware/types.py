"""Types for middleware and agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeGuard,
    cast,
    overload,
)
from collections import defaultdict

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

JumpTo = Literal["tools", "model", "__end__"]
"""Destination to jump to when a middleware node returns."""

ResponseT = TypeVar("ResponseT")


@dataclass
class ModelRequest:
    """Model request information for the agent."""

    model: BaseChatModel
    system_prompt: str | None
    messages: list[AnyMessage]  # excluding system prompt
    tool_choice: Any | None
    tools: list[BaseTool]
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


class JumpsMap(TypedDict):
    """Valid jump configuration for a middleware."""

    before_model: list[JumpTo]
    after_model: list[JumpTo]


class AgentMiddleware(Generic[StateT, ContextT]):
    """Base middleware class for an agent.

    Subclass this and implement any of the defined methods to customize agent behavior
    between steps in the main agent loop.
    """

    state_schema: type[StateT] = cast("type[StateT]", AgentState)
    """The schema for state passed to the middleware nodes."""

    tools: list[BaseTool]
    """Additional tools registered by the middleware."""

    jumps_map: JumpsMap = {"before_model": [], "after_model": []}
    """Jumps associated with the middleware's hooks. Used to establish conditional edges."""

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


class _CallableWithState(Protocol):
    """Callable with AgentState as argument."""

    def __call__(self, state: AgentState) -> dict[str, Any] | Command | None:
        """Perform some logic with the state."""
        ...


class _CallableWithStateAndRuntime(Protocol[ContextT]):
    """Callable with AgentState and Runtime as arguments."""

    def __call__(
        self, state: AgentState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | Command | None:
        """Perform some logic with the state and runtime."""
        ...


class _CallableWithModelRequestAndState(Protocol):
    """Callable with ModelRequest and AgentState as arguments."""

    def __call__(self, request: ModelRequest, state: AgentState) -> ModelRequest:
        """Perform some logic with the model request and state."""
        ...


class _CallableWithModelRequestAndStateAndRuntime(Protocol[ContextT]):
    """Callable with ModelRequest, AgentState, and Runtime as arguments."""

    def __call__(
        self, request: ModelRequest, state: AgentState, runtime: Runtime[ContextT]
    ) -> ModelRequest:
        """Perform some logic with the model request, state, and runtime."""
        ...


_NodeSignature: TypeAlias = _CallableWithState | _CallableWithStateAndRuntime[ContextT]
_ModelRequestSignature: TypeAlias = (
    _CallableWithModelRequestAndState | _CallableWithModelRequestAndStateAndRuntime[ContextT]
)


def is_callable_with_runtime(
    func: _NodeSignature[ContextT],
) -> TypeGuard[_CallableWithStateAndRuntime[ContextT]]:
    return "runtime" in signature(func).parameters


def is_callable_with_runtime_and_request(
    func: _ModelRequestSignature[ContextT],
) -> TypeGuard[_CallableWithModelRequestAndStateAndRuntime[ContextT]]:
    return "runtime" in signature(func).parameters


@overload
def before_model(
    func: _NodeSignature[ContextT],
) -> AgentMiddleware[AgentState, ContextT]: ...


@overload
def before_model(
    func: None = None,
    *,
    state_schema: type[StateT] = AgentState,
    tools: list[BaseTool] | None = None,
    jumps: list[JumpTo] | None = None,
    name: str = "BeforeModelMiddleware",
) -> Callable[[_NodeSignature[ContextT]], AgentMiddleware[StateT, ContextT]]: ...


def before_model(
    func: _NodeSignature[ContextT] | None = None,
    *,
    state_schema: type[StateT] = AgentState,
    tools: list[BaseTool] | None = None,
    jumps: list[JumpTo] | None = None,
    name: str = "BeforeModelMiddleware",
) -> (
    Callable[[_NodeSignature[ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the before_model hook."""

    def decorator(func: _NodeSignature[ContextT]) -> AgentMiddleware[StateT, ContextT]:
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
                return func(state)

            wrapped = wrapped_without_runtime

        return type(
            name,
            (AgentMiddleware[StateT, ContextT],),
            {
                "state_schema": state_schema,
                "tools": tools or [],
                "jumps_map": {"before_model": jumps},
                "before_model": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def modify_model_request(
    func: _ModelRequestSignature[ContextT],
) -> AgentMiddleware[AgentState, ContextT]: ...


@overload
def modify_model_request(
    func: None = None,
    *,
    state_schema: type[StateT] = AgentState,
    tools: list[BaseTool] | None = None,
    jumps: list[JumpTo] | None = None,
    name: str = "ModifyModelRequestMiddleware",
) -> Callable[[_ModelRequestSignature[ContextT]], AgentMiddleware[StateT, ContextT]]: ...


def modify_model_request(
    func: _ModelRequestSignature[ContextT] | None = None,
    *,
    state_schema: type[StateT] = AgentState,
    tools: list[BaseTool] | None = None,
    jumps: list[JumpTo] | None = None,
    name: str = "ModifyModelRequestMiddleware",
) -> (
    Callable[[_ModelRequestSignature[ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the modify_model_request hook."""

    def decorator(func: _ModelRequestSignature[ContextT]) -> AgentMiddleware[StateT, ContextT]:
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
                return func(request, state)

            wrapped = wrapped_without_runtime

        return type(
            name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema,
                "tools": tools or [],
                "jumps_map": {"modify_model_request": jumps},
                "modify_model_request": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def after_model(
    func: _NodeSignature[ContextT],
) -> AgentMiddleware[AgentState, ContextT]: ...


@overload
def after_model(
    func: None = None,
    *,
    state_schema: type[StateT] = AgentState,
    tools: list[BaseTool] | None = None,
    jumps: list[JumpTo] | None = None,
    name: str = "AfterModelMiddleware",
) -> Callable[[_NodeSignature[ContextT]], AgentMiddleware[StateT, ContextT]]: ...


def after_model(
    func: _NodeSignature[ContextT] | None = None,
    *,
    state_schema: type[StateT] = AgentState,
    tools: list[BaseTool] | None = None,
    jumps: list[JumpTo] | None = None,
    name: str = "AfterModelMiddleware",
) -> (
    Callable[[_NodeSignature[ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the after_model hook."""

    def decorator(func: _NodeSignature[ContextT]) -> AgentMiddleware[StateT, ContextT]:
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
                return func(state)

            wrapped = wrapped_without_runtime

        return type(
            name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema,
                "tools": tools or [],
                "jumps_map": {"after_model": jumps},
                "after_model": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator
