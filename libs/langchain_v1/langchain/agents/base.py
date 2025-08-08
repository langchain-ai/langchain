from langchain_core.tools.base import BaseTool
from typing_extensions import Unpack, TypedDict, ParamSpec
from typing import (
    TYPE_CHECKING,
    Concatenate,
    TypeAlias,
    Callable,
    TypeVar,
    Awaitable,
    Any,
    Sequence,
    Generic,
    cast,
    overload,
)
from langgraph.runtime import Runtime
from langchain_core.v1.messages import MessageV1, SystemMessage
from pydantic import BaseModel
from langchain_core.runnables.base import RunnableLike
from langgraph.graph import StateGraph
from langgraph.typing import ContextT

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore
    from langgraph.types import Checkpointer

from langchain_core.v1.chat_models import BaseChatModel

from langchain.agents._utils import DeprecatedKwargs

ResponseT = TypeVar("ResponseT", bound=dict | BaseModel | None, default=None)


class AgentState(TypedDict, Generic[ResponseT]):
    messages: list[MessageV1]
    remaining_steps: int
    structured_response: ResponseT


AgentStateT = TypeVar("AgentStateT", bound=AgentState)

DynamicModel: TypeAlias = Callable[
    [AgentStateT, Runtime[ContextT]], BaseChatModel | Awaitable[BaseChatModel]
]
Model: TypeAlias = str | BaseChatModel | DynamicModel

ToolParams = ParamSpec("ToolParams", default=...)
ToolFunctionPlain = Callable[ToolParams, Any]
ToolFunctionState = Callable[Concatenate[AgentStateT, ToolParams], Any]
ToolFunctionContext = Callable[
    Concatenate[AgentStateT, Runtime[ContextT], ToolParams], Any
]

ToolFunction: TypeAlias = ToolFunctionPlain | ToolFunctionState | ToolFunctionContext

BuiltinTool: TypeAlias = dict[str, Any]
Tool: TypeAlias = BaseTool | ToolFunction | BuiltinTool

MessageOrDict: TypeAlias = MessageV1 | dict[str, Any]
Prompt: TypeAlias = (
    str | SystemMessage | Callable[[AgentStateT], Sequence[MessageOrDict]]
)


class ToolNode:
    """Backfill for now."""


class _AgentBuilder(Generic[AgentStateT, ContextT, ResponseT]):
    state_schema: type[AgentStateT]
    response_schema: type[ResponseT]

    def __init__(
        self,
        model: Model,
        tools: Sequence[Tool] | ToolNode | None,
        prompt: Prompt | None,
        response_format: tuple[str, type[ResponseT]] | type[ResponseT] | None,
        pre_model_hook: RunnableLike | None,
        post_model_hook: RunnableLike | None,
        state_schema: type[AgentStateT],
        context_schema: type[ContextT] | None,
    ):
        self.model = model
        self.tools = tools
        self.prompt = prompt
        self.response_format = response_format
        self.pre_model_hook = pre_model_hook
        self.post_model_hook = post_model_hook
        self.state_schema = state_schema
        self.context_schema = context_schema

        if isinstance(response_format, tuple):
            self.response_schema = response_format[1]
        elif response_format is not None:
            self.response_schema = response_format

    def build(self) -> StateGraph[AgentStateT, ContextT]:
        """Dynamically build the graph associated with the agent specifications."""

        workflow = StateGraph(
            state_schema=self.state_schema, context_schema=self.context_schema
        )

        return workflow


def create_agent(
    model: Model,
    tools: Sequence[Tool] | ToolNode | None = None,
    prompt: Prompt | None = None,
    response_format: tuple[str, type[ResponseT]] | type[ResponseT] | None = None,
    pre_model_hook: RunnableLike | None = None,
    post_model_hook: RunnableLike | None = None,
    state_schema: type[AgentStateT] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    name: str | None = None,
    **deprecated_kwargs: Unpack[DeprecatedKwargs],
) -> CompiledStateGraph[AgentStateT, ContextT]:
    resolved_state_schema = cast(type[AgentStateT], state_schema or AgentState)

    builder = _AgentBuilder[AgentStateT, ContextT, ResponseT](
        model=model,
        tools=tools,
        prompt=prompt,
        response_format=response_format,
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
        state_schema=resolved_state_schema,
        context_schema=context_schema,
    )

    workflow = builder.build()
    agent = workflow.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        name=name,
    )

    return agent
