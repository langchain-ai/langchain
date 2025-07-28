"""**Messages** are objects used in prompts and chat conversations.

**Class hierarchy:**

.. code-block::

    BaseMessage --> SystemMessage, AIMessage, HumanMessage, ChatMessage, FunctionMessage, ToolMessage
                --> BaseMessageChunk --> SystemMessageChunk, AIMessageChunk, HumanMessageChunk, ChatMessageChunk, FunctionMessageChunk, ToolMessageChunk

**Main helpers:**

.. code-block::

    ChatPromptTemplate

"""  # noqa: E501

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

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
        Annotation,
        AudioContentBlock,
        Citation,
        CodeInterpreterCall,
        CodeInterpreterOutput,
        CodeInterpreterResult,
        ContentBlock,
        DataContentBlock,
        FileContentBlock,
        ImageContentBlock,
        NonStandardAnnotation,
        NonStandardContentBlock,
        PlainTextContentBlock,
        ReasoningContentBlock,
        TextContentBlock,
        VideoContentBlock,
        WebSearchCall,
        WebSearchResult,
        convert_to_openai_data_block,
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

__all__ = (
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
    "CodeInterpreterCall",
    "CodeInterpreterOutput",
    "CodeInterpreterResult",
    "ContentBlock",
    "DataContentBlock",
    "FileContentBlock",
    "FunctionMessage",
    "FunctionMessageChunk",
    "HumanMessage",
    "HumanMessageChunk",
    "ImageContentBlock",
    "InvalidToolCall",
    "MessageLikeRepresentation",
    "NonStandardAnnotation",
    "NonStandardContentBlock",
    "PlainTextContentBlock",
    "ReasoningContentBlock",
    "RemoveMessage",
    "SystemMessage",
    "SystemMessageChunk",
    "TextContentBlock",
    "ToolCall",
    "ToolCallChunk",
    "ToolMessage",
    "ToolMessageChunk",
    "VideoContentBlock",
    "WebSearchCall",
    "WebSearchResult",
    "_message_from_dict",
    "convert_to_messages",
    "convert_to_openai_data_block",
    "convert_to_openai_image_block",
    "convert_to_openai_messages",
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
    "Annotation": "content_blocks",
    "AudioContentBlock": "content_blocks",
    "BaseMessage": "base",
    "BaseMessageChunk": "base",
    "merge_content": "base",
    "message_to_dict": "base",
    "messages_to_dict": "base",
    "Citation": "content_blocks",
    "ContentBlock": "content_blocks",
    "ChatMessage": "chat",
    "ChatMessageChunk": "chat",
    "CodeInterpreterCall": "content_blocks",
    "CodeInterpreterOutput": "content_blocks",
    "CodeInterpreterResult": "content_blocks",
    "DataContentBlock": "content_blocks",
    "FileContentBlock": "content_blocks",
    "FunctionMessage": "function",
    "FunctionMessageChunk": "function",
    "HumanMessage": "human",
    "HumanMessageChunk": "human",
    "NonStandardAnnotation": "content_blocks",
    "NonStandardContentBlock": "content_blocks",
    "PlainTextContentBlock": "content_blocks",
    "ReasoningContentBlock": "content_blocks",
    "RemoveMessage": "modifier",
    "SystemMessage": "system",
    "SystemMessageChunk": "system",
    "WebSearchCall": "content_blocks",
    "WebSearchResult": "content_blocks",
    "ImageContentBlock": "content_blocks",
    "InvalidToolCall": "tool",
    "TextContentBlock": "content_blocks",
    "ToolCall": "tool",
    "ToolCallChunk": "tool",
    "ToolMessage": "tool",
    "ToolMessageChunk": "tool",
    "VideoContentBlock": "content_blocks",
    "AnyMessage": "utils",
    "MessageLikeRepresentation": "utils",
    "_message_from_dict": "utils",
    "convert_to_messages": "utils",
    "convert_to_openai_data_block": "content_blocks",
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
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
