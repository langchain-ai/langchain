"""**Messages** are objects used in prompts and chat conversations."""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr
from langchain_core.utils.utils import LC_AUTO_PREFIX, LC_ID_PREFIX, ensure_id

if TYPE_CHECKING:
    from langchain_core.messages.ai import (
        AIMessage,
        AIMessageChunk,
        InputTokenDetails,
        OutputTokenDetails,
        UsageMetadata,
    )
    from langchain_core.messages.base import (
        BaseMessage,
        BaseMessageChunk,
        merge_content,
        message_to_dict,
        messages_to_dict,
    )
    from langchain_core.messages.block_translators.openai import (
        convert_to_openai_data_block,
        convert_to_openai_image_block,
    )
    from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
    from langchain_core.messages.content import (
        Annotation,
        AudioContentBlock,
        Citation,
        ContentBlock,
        DataContentBlock,
        FileContentBlock,
        ImageContentBlock,
        InvalidToolCall,
        NonStandardAnnotation,
        NonStandardContentBlock,
        PlainTextContentBlock,
        ReasoningContentBlock,
        ServerToolCall,
        ServerToolCallChunk,
        ServerToolResult,
        TextContentBlock,
        VideoContentBlock,
        is_data_content_block,
    )
    from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
    from langchain_core.messages.human import HumanMessage, HumanMessageChunk
    from langchain_core.messages.modifier import RemoveMessage
    from langchain_core.messages.system import SystemMessage, SystemMessageChunk
    from langchain_core.messages.tool import (
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

__all__ = (
    "LC_AUTO_PREFIX",
    "LC_ID_PREFIX",
    "AIMessage",
    "AIMessageChunk",
    "Annotation",
    "AnyMessage",
    "AudioContentBlock",
    "BaseMessage",
    "BaseMessageChunk",
    "ChatMessage",
    "ChatMessageChunk",
    "Citation",
    "ContentBlock",
    "DataContentBlock",
    "FileContentBlock",
    "FunctionMessage",
    "FunctionMessageChunk",
    "HumanMessage",
    "HumanMessageChunk",
    "ImageContentBlock",
    "InputTokenDetails",
    "InvalidToolCall",
    "MessageLikeRepresentation",
    "NonStandardAnnotation",
    "NonStandardContentBlock",
    "OutputTokenDetails",
    "PlainTextContentBlock",
    "ReasoningContentBlock",
    "RemoveMessage",
    "ServerToolCall",
    "ServerToolCallChunk",
    "ServerToolResult",
    "SystemMessage",
    "SystemMessageChunk",
    "TextContentBlock",
    "ToolCall",
    "ToolCallChunk",
    "ToolMessage",
    "ToolMessageChunk",
    "UsageMetadata",
    "VideoContentBlock",
    "_message_from_dict",
    "convert_to_messages",
    "convert_to_openai_data_block",
    "convert_to_openai_image_block",
    "convert_to_openai_messages",
    "ensure_id",
    "filter_messages",
    "get_buffer_string",
    "is_data_content_block",
    "merge_content",
    "merge_message_runs",
    "message_chunk_to_message",
    "message_to_dict",
    "messages_from_dict",
    "messages_to_dict",
    "trim_messages",
)

_dynamic_imports = {
    "AIMessage": "ai",
    "AIMessageChunk": "ai",
    "Annotation": "content",
    "AudioContentBlock": "content",
    "BaseMessage": "base",
    "BaseMessageChunk": "base",
    "merge_content": "base",
    "message_to_dict": "base",
    "messages_to_dict": "base",
    "Citation": "content",
    "ContentBlock": "content",
    "ChatMessage": "chat",
    "ChatMessageChunk": "chat",
    "DataContentBlock": "content",
    "FileContentBlock": "content",
    "FunctionMessage": "function",
    "FunctionMessageChunk": "function",
    "HumanMessage": "human",
    "HumanMessageChunk": "human",
    "NonStandardAnnotation": "content",
    "NonStandardContentBlock": "content",
    "OutputTokenDetails": "ai",
    "PlainTextContentBlock": "content",
    "ReasoningContentBlock": "content",
    "RemoveMessage": "modifier",
    "ServerToolCall": "content",
    "ServerToolCallChunk": "content",
    "ServerToolResult": "content",
    "SystemMessage": "system",
    "SystemMessageChunk": "system",
    "ImageContentBlock": "content",
    "InputTokenDetails": "ai",
    "InvalidToolCall": "tool",
    "TextContentBlock": "content",
    "ToolCall": "tool",
    "ToolCallChunk": "tool",
    "ToolMessage": "tool",
    "ToolMessageChunk": "tool",
    "UsageMetadata": "ai",
    "VideoContentBlock": "content",
    "AnyMessage": "utils",
    "MessageLikeRepresentation": "utils",
    "_message_from_dict": "utils",
    "convert_to_messages": "utils",
    "convert_to_openai_data_block": "block_translators.openai",
    "convert_to_openai_image_block": "block_translators.openai",
    "convert_to_openai_messages": "utils",
    "filter_messages": "utils",
    "get_buffer_string": "utils",
    "is_data_content_block": "content",
    "merge_message_runs": "utils",
    "message_chunk_to_message": "utils",
    "messages_from_dict": "utils",
    "trim_messages": "utils",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
