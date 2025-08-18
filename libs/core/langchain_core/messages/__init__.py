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
        LC_AUTO_PREFIX,
        LC_ID_PREFIX,
        BaseMessage,
        BaseMessageChunk,
        ensure_id,
        merge_content,
        message_to_dict,
        messages_to_dict,
    )
    from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
    from langchain_core.messages.content import (
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
        is_reasoning_block,
        is_text_block,
        is_tool_call_block,
        is_tool_call_chunk,
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
    "ensure_id",
    "filter_messages",
    "get_buffer_string",
    "is_data_content_block",
    "is_reasoning_block",
    "is_text_block",
    "is_tool_call_block",
    "is_tool_call_chunk",
    "merge_content",
    "merge_message_runs",
    "message_chunk_to_message",
    "message_to_dict",
    "messages_from_dict",
    "messages_to_dict",
    "trim_messages",
)

_dynamic_imports = {
    "ensure_id": "base",
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
    "CodeInterpreterCall": "content",
    "CodeInterpreterOutput": "content",
    "CodeInterpreterResult": "content",
    "DataContentBlock": "content",
    "FileContentBlock": "content",
    "FunctionMessage": "function",
    "FunctionMessageChunk": "function",
    "HumanMessage": "human",
    "HumanMessageChunk": "human",
    "LC_AUTO_PREFIX": "base",
    "LC_ID_PREFIX": "base",
    "NonStandardAnnotation": "content",
    "NonStandardContentBlock": "content",
    "PlainTextContentBlock": "content",
    "ReasoningContentBlock": "content",
    "RemoveMessage": "modifier",
    "SystemMessage": "system",
    "SystemMessageChunk": "system",
    "WebSearchCall": "content",
    "WebSearchResult": "content",
    "ImageContentBlock": "content",
    "InvalidToolCall": "tool",
    "TextContentBlock": "content",
    "ToolCall": "tool",
    "ToolCallChunk": "tool",
    "ToolMessage": "tool",
    "ToolMessageChunk": "tool",
    "VideoContentBlock": "content",
    "AnyMessage": "utils",
    "MessageLikeRepresentation": "utils",
    "_message_from_dict": "utils",
    "convert_to_messages": "utils",
    "convert_to_openai_data_block": "content",
    "convert_to_openai_image_block": "content",
    "convert_to_openai_messages": "utils",
    "filter_messages": "utils",
    "get_buffer_string": "utils",
    "is_data_content_block": "content",
    "is_reasoning_block": "content",
    "is_text_block": "content",
    "is_tool_call_block": "content",
    "is_tool_call_chunk": "content",
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
