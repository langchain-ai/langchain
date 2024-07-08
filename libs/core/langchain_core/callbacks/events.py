from typing import Optional, Dict, Literal, List
from typing import TypedDict, Any, Union, Callable

from tenacity import RetryCallState

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import GenerationChunk, LLMResult, ChatGenerationChunk


class RetrieverErrorEvent(TypedDict):
    type: Literal["on_retriever_error"]
    error: BaseException


class RetrieverEndEvent(TypedDict):
    type: Literal["on_retriever_end"]
    documents: List[Document]


class LLMNewTokenEvent(TypedDict):
    type: Literal["on_llm_new_token"]
    token: str
    chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]]


class LLMEndEvent(TypedDict):
    type: Literal["on_llm_end"]
    response: LLMResult


class LLMErrorEvent(TypedDict):
    type: Literal["on_llm_error"]
    error: BaseException


class ChainEndEvent(TypedDict):
    type: Literal["on_chain_end"]
    outputs: Dict[str, Any]


class ChainErrorEvent(TypedDict):
    """Event for a chain error."""

    type: Literal["on_chain_error"]
    error: BaseException


class AgentActionEvent(TypedDict):
    """Event for an agent action."""

    type: Literal["on_agent_action"]
    action: AgentAction


class AgentFinishEvent(TypedDict):
    """Event for an agent action."""

    type: Literal["on_agent_finish"]
    finish: AgentFinish


class ToolEndEvent(TypedDict):
    """Event for a tool end."""

    type: Literal["on_tool_end"]
    output: Any


class ToolErrorEvent(TypedDict):
    """Event for a tool error."""

    type: Literal["on_tool_error"]
    error: BaseException


class LLMStartEvent(TypedDict):
    type: Literal["on_llm_start"]
    serialized: Dict[str, Any]
    prompts: List[str]


class ChatModelStartEvent(TypedDict):
    type: Literal["on_chat_model_start"]
    serialized: Dict[str, Any]
    messages: List[List[BaseMessage]]


class AdHocEvent(TypedDict):
    """Ad hoc event."""
    type: Literal["on_ad_hoc"]
    data: Any


class RetrieverStartEvent(TypedDict):
    type: Literal["on_retriever_start"]
    serialized: Dict[str, Any]
    query: str


class ChainStartEvent(TypedDict):
    type: Literal["on_chain_start"]
    serialized: Dict[str, Any]
    inputs: Dict[str, Any]


class ToolStartEvent(TypedDict):
    type: Literal["on_tool_start"]
    serialized: Dict[str, Any]
    input_str: str
    inputs: Optional[Dict[str, Any]]


class TextEvent(TypedDict):
    type: Literal["on_text"]
    text: str


class RetryEvent(TypedDict):
    type: Literal["on_retry"]
    retry_state: RetryCallState


# define possible callback events
Event = Union[
    RetrieverErrorEvent,
    RetrieverEndEvent,
    LLMNewTokenEvent,
    LLMEndEvent,
    LLMErrorEvent,
    ChainEndEvent,
    ChainErrorEvent,
    AgentActionEvent,
    AgentFinishEvent,
    ToolEndEvent,
    ToolErrorEvent,
    LLMStartEvent,
    ChatModelStartEvent,
    RetrieverStartEvent,
    ChainStartEvent,
    ToolStartEvent,
    TextEvent,
    RetryEvent,
]


class Handlers(TypedDict, total=False):
    on_chain_error: Union[Callable[[ChainErrorEvent], Any]]
    on_chain_start: Union[Callable[[ChainStartEvent], Any]]


def func(inputs: Any, callbacks: Optional[Handlers]):
    pass


def _on_chain_error(event: ChainErrorEvent):
    pass


def zoo(inputs: Any):
    func(inputs, callbacks={"on_chain_error": _on_chain_error})
