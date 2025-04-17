"""**Messages** are objects used in prompts and chat conversations.

**Class hierarchy:**

.. code-block::

    BaseMessage --> SystemMessage, AIMessage, HumanMessage, ChatMessage, FunctionMessage, ToolMessage
                --> BaseMessageChunk --> SystemMessageChunk, AIMessageChunk, HumanMessageChunk, ChatMessageChunk, FunctionMessageChunk, ToolMessageChunk

**Main helpers:**

.. code-block::

    ChatPromptTemplate

"""  # noqa: E501

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages.ai import (
        AIMessage,
        AIMessageChunk,
    )
    from langchain_core.messages.base import (
        BaseMessage,
        BaseMessageChunk,
        merge_content,
        message_to_dict,
        messages_to_dict,
    )
    from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
    from langchain_core.messages.content_blocks import (
        convert_to_openai_image_block,
        is_data_content_block,
    )
    from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
    from langchain_core.messages.human import HumanMessage, HumanMessageChunk
    from langchain_core.messages.modifier import RemoveMessage
    from langchain_core.messages.system import SystemMessage, SystemMessageChunk
    from langchain_core.messages.tool import (
        InvalidToolCall,
        ToolCall,
        ToolCallChunk,
        ToolMessage,
        ToolMessageChunk,
    )
    from langchain_core.messages.utils import (
        AnyMessage,
        MessageLikeRepresentation,
        _message_from_dict,
        convert_to_messages,
        convert_to_openai_messages,
        filter_messages,
        get_buffer_string,
        merge_message_runs,
        message_chunk_to_message,
        messages_from_dict,
        trim_messages,
    )

__all__ = [
    "AIMessage",
    "AIMessageChunk",
    "AnyMessage",
    "BaseMessage",
    "BaseMessageChunk",
    "ChatMessage",
    "ChatMessageChunk",
    "FunctionMessage",
    "FunctionMessageChunk",
    "HumanMessage",
    "HumanMessageChunk",
    "InvalidToolCall",
    "MessageLikeRepresentation",
    "SystemMessage",
    "SystemMessageChunk",
    "ToolCall",
    "ToolCallChunk",
    "ToolMessage",
    "ToolMessageChunk",
    "RemoveMessage",
    "_message_from_dict",
    "convert_to_openai_image_block",
    "convert_to_messages",
    "get_buffer_string",
    "is_data_content_block",
    "merge_content",
    "message_chunk_to_message",
    "message_to_dict",
    "messages_from_dict",
    "messages_to_dict",
    "filter_messages",
    "merge_message_runs",
    "trim_messages",
    "convert_to_openai_messages",
]

_dynamic_imports = {
    "AIMessage": "ai",
    "AIMessageChunk": "ai",
    "BaseMessage": "base",
    "BaseMessageChunk": "base",
    "merge_content": "base",
    "message_to_dict": "base",
    "messages_to_dict": "base",
    "ChatMessage": "chat",
    "ChatMessageChunk": "chat",
    "FunctionMessage": "function",
    "FunctionMessageChunk": "function",
    "HumanMessage": "human",
    "HumanMessageChunk": "human",
    "RemoveMessage": "modifier",
    "SystemMessage": "system",
    "SystemMessageChunk": "system",
    "InvalidToolCall": "tool",
    "ToolCall": "tool",
    "ToolCallChunk": "tool",
    "ToolMessage": "tool",
    "ToolMessageChunk": "tool",
    "AnyMessage": "utils",
    "MessageLikeRepresentation": "utils",
    "_message_from_dict": "utils",
    "convert_to_messages": "utils",
    "convert_to_openai_image_block": "content_blocks",
    "convert_to_openai_messages": "utils",
    "filter_messages": "utils",
    "get_buffer_string": "utils",
    "is_data_content_block": "content_blocks",
    "merge_message_runs": "utils",
    "message_chunk_to_message": "utils",
    "messages_from_dict": "utils",
    "trim_messages": "utils",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    package = __spec__.parent
    if module_name == "__module__" or module_name is None:
        result = import_module(f".{attr_name}", package=package)
    else:
        module = import_module(f".{module_name}", package=package)
        result = getattr(module, attr_name)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
