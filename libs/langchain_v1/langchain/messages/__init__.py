"""Message and message content types.

Includes message types for different roles (e.g., human, AI, system), as well as types
for message content blocks (e.g., text, image, audio) and tool calls.

!!! warning "Reference docs"
    This page contains **reference documentation** for Messages. See
    [the docs](https://docs.langchain.com/oss/python/langchain/messages) for conceptual
    guides, tutorials, and examples on using Messages.
"""

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    Annotation,
    AnyMessage,
    AudioContentBlock,
    Citation,
    ContentBlock,
    DataContentBlock,
    FileContentBlock,
    HumanMessage,
    ImageContentBlock,
    InputTokenDetails,
    InvalidToolCall,
    MessageLikeRepresentation,
    NonStandardAnnotation,
    NonStandardContentBlock,
    OutputTokenDetails,
    PlainTextContentBlock,
    ReasoningContentBlock,
    RemoveMessage,
    ServerToolCall,
    ServerToolCallChunk,
    ServerToolResult,
    SystemMessage,
    TextContentBlock,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
    UsageMetadata,
    VideoContentBlock,
    trim_messages,
)

__all__ = [
    "AIMessage",
    "AIMessageChunk",
    "Annotation",
    "AnyMessage",
    "AudioContentBlock",
    "Citation",
    "ContentBlock",
    "DataContentBlock",
    "FileContentBlock",
    "HumanMessage",
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
    "TextContentBlock",
    "ToolCall",
    "ToolCallChunk",
    "ToolMessage",
    "UsageMetadata",
    "VideoContentBlock",
    "trim_messages",
]
