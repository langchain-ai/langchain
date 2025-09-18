"""Types for middleware and agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, cast

# needed as top level import for pydantic schema generation on AgentState
from langchain_core.messages import AnyMessage  # noqa: TC002
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph.message import Messages, add_messages
from langgraph.runtime import Runtime
from langgraph.typing import ContextT
from typing_extensions import NotRequired, Required, TypedDict, TypeVar

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime

    from langchain.agents.structured_output import ResponseFormat

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


class AgentState(TypedDict, Generic[ResponseT]):
    """State schema for the agent."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue]]
    response: NotRequired[ResponseT]


class PublicAgentState(TypedDict, Generic[ResponseT]):
    """Input / output schema for the agent."""

    messages: Required[Messages]
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
