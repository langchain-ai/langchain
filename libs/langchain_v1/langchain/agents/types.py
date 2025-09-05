from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Generic, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph.message import Messages, add_messages
from typing_extensions import TypedDict, TypeVar

from langchain.agents.structured_output import ResponseFormat

JumpTo = Literal["tools", "model", "__end__"]
ResponseT = TypeVar("ResponseT")


@dataclass
class ModelRequest:
    model: BaseChatModel
    system_prompt: str
    messages: list[AnyMessage]  # excluding system prompt
    tool_choice: Any
    tools: list[BaseTool]
    response_format: ResponseFormat | None


class AgentState(TypedDict, Generic[ResponseT], total=False):
    # TODO: import change allowing for required / not required and still registering reducer properly
    # do we want to use total = False or require NotRequired?
    messages: Annotated[list[AnyMessage], add_messages]
    model_request: Annotated[ModelRequest | None, EphemeralValue]
    jump_to: Annotated[JumpTo | None, EphemeralValue]

    # TODO: structured response maybe?
    response: ResponseT


StateT = TypeVar("StateT", bound=AgentState)


class AgentMiddleware(Generic[StateT]):
    # TODO: I thought this should be a ClassVar[type[StateT]] but inherently class vars can't use type vars
    # bc they're instance dependent
    state_schema: ClassVar[type] = AgentState
    tools: list[BaseTool] = []

    def before_model(self, state: StateT) -> AgentUpdate | AgentJump | None:
        pass

    def modify_model_request(self, request: ModelRequest, state: StateT) -> ModelRequest:
        return request

    def after_model(self, state: StateT) -> AgentUpdate | AgentJump | None:
        pass


class AgentUpdate(TypedDict, total=False):
    messages: Messages
    response: dict


class AgentJump(TypedDict, total=False):
    messages: Messages
    jump_to: JumpTo
