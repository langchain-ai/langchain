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
)
from langgraph.runtime import Runtime
from langchain_core.v1.messages import MessageV1, SystemMessage
from pydantic import BaseModel
from langchain_core.runnables.base import RunnableLike

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore
    from langgraph.types import Checkpointer
    from langgraph.typing import ContextT

from langchain_core.v1.chat_models import BaseChatModel

from langchain.agents._utils import DeprecatedKwargs


class AgentState(TypedDict):
    messages: list[MessageV1]
    remaining_steps: int


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

Prompt: TypeAlias = str | SystemMessage | Callable[[AgentStateT], str | SystemMessage]

ResponseT = TypeVar("ResponseT", bound=dict | BaseModel)
ResponseFormat: TypeAlias = tuple[str, type[ResponseT]] | type[ResponseT]


def create_agent(
    model: Model,
    tools: Sequence[Tool] | ToolNode | None = None,
    prompt: Prompt | None = None,
    response_format: ResponseFormat | None = None,
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
) -> CompiledStateGraph: ...
