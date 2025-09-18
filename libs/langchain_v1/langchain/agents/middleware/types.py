"""Types for middleware and agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, cast

# needed as top level import for pydantic schema generation on AgentState
from langchain_core.messages import AnyMessage  # noqa: TC002
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
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


class AgentMiddleware(Generic[StateT, ContextT]):
    """Base middleware class for an agent.

    Subclass this and implement any of the defined methods to customize agent behavior
    between steps in the main agent loop.
    """

    state_schema: type[StateT] = cast("type[StateT]", AgentState)
    """The schema for state passed to the middleware nodes."""

    tools: list[BaseTool]
    """Additional tools registered by the middleware."""

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
