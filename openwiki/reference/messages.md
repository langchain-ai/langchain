---
type: "API Reference"
title: "Messages API Reference"
description: "Complete API reference for LangChain message types, serialization utilities, content blocks, and notification metadata functions for marking system-generated messages in conversations."
tags: ["messages", "api", "serialization", "notification-metadata", "content-blocks"]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-30T08:19:17.669Z
sources:
  - id: openwiki-source-85004ebd4eba7aec1e145503
    resource: repo://libs/langchain/langchain_classic/schema/messages.py
generated: { by: "openwiki/0.5.0", at: "2026-10-30T08:19:17.669Z" }
---

# Messages API Reference

This reference documents the public APIs for working with messages in LangChain: message types, serialization/deserialization functions, content block structures, utility functions for filtering and merging, and notification metadata for marking system-generated messages.

## Overview

Messages are the fundamental I/O abstraction for language model interactions. The API provides:

- **Message types**: `BaseMessage`, `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`, `ChatMessage`, `FunctionMessage`, `RemoveMessage`, and streaming variants
- **Serialization**: `message_to_dict()`, `messages_to_dict()`, `messages_from_dict()`, `_message_from_dict()`
- **Content blocks**: Standardized multimodal content (text, images, audio, video, files, tool calls, reasoning)
- **Utilities**: `filter_messages()`, `merge_message_runs()`, `message_chunk_to_message()`, `trim_messages()`, `get_buffer_string()`
- **Notification metadata**: `NotificationMetadata`, `add_notification_metadata()`, `get_notification_metadata()` for marking agent-added messages

## Core Import

All public message APIs are available from `langchain_core.messages`:

```python
from langchain_core.messages import (
    # Message types
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage,
    ChatMessage, FunctionMessage, RemoveMessage,
    # Chunks
    BaseMessageChunk, HumanMessageChunk, AIMessageChunk, SystemMessageChunk,
    ToolMessageChunk, ChatMessageChunk, FunctionMessageChunk,
    # Serialization
    message_to_dict, messages_to_dict, messages_from_dict,
    # Content blocks
    ContentBlock, TextContentBlock, ImageContentBlock, AudioContentBlock,
    VideoContentBlock, FileContentBlock, DataContentBlock,
    # Tool calling
    ToolCall, ToolCallChunk, InvalidToolCall,
    # Notification metadata
    NotificationMetadata, add_notification_metadata, get_notification_metadata,
    # Utilities
    filter_messages, merge_message_runs, message_chunk_to_message,
    trim_messages, get_buffer_string, AnyMessage,
)
```

## Message Types

All message types inherit from `BaseMessage` and are serializable. For detailed semantics, see [`/openwiki/concepts/messages.md`](/openwiki/concepts/messages.md).

### BaseMessage

The abstract base class for all messages. Defines the common contract:

```python
class BaseMessage:
    content: str | list[str | dict[Any, Any]]
    """Message payload: plain string, or list of content blocks."""
    
    type: str
    """Discriminator for serialization (e.g., 'human', 'ai', 'system', 'tool')."""
    
    id: str | None
    """Optional unique identifier, typically assigned by the provider."""
    
    name: str | None
    """Optional speaker/function name."""
    
    additional_kwargs: dict[Any, Any]
    """Provider-specific metadata (not sent to models by integrations)."""
    
    response_metadata: dict[Any, Any]
    """Headers, token usage, model name, finish reason, logprobs, etc."""
```

**Key methods and properties**:

- `is_lc_serializable()` → `True`: Messages are always serializable
- `.text` property: Extracts text content from both string and list-based content
- `.content_blocks` property: Parses message content into typed `ContentBlock` objects
- `model_copy(update={...})`: Create a shallow copy with updated fields
- `model_dump()`: Convert to a plain dict for serialization

### HumanMessage

User input to the model.

```python
from langchain_core.messages import HumanMessage

msg = HumanMessage(content="What is the capital of France?")

# Multimodal
msg = HumanMessage(content=[
    {"type": "text", "text": "Describe this image"},
    {"type": "image", "url": "https://example.com/image.png"}
])
```

- **`type`**: Always `"human"` (discriminator)
- **Streaming variant**: `HumanMessageChunk`

### AIMessage

Model response, optionally with tool calls.

```python
from langchain_core.messages import AIMessage

# Plain text
msg = AIMessage(content="The capital of France is Paris.")

# With tool calls
msg = AIMessage(
    content="I'll search for that.",
    tool_calls=[
        {
            "type": "tool_call",
            "id": "call_123",
            "name": "search",
            "args": {"query": "Python tutorials"}
        }
    ]
)

# With token usage
msg = AIMessage(
    content="...",
    usage_metadata={
        "input_tokens": 350,
        "output_tokens": 240,
        "total_tokens": 590
    }
)
```

**Unique fields**:
- **`tool_calls`**: `list[ToolCall]` — Tool invocations requested by the model
- **`invalid_tool_calls`**: `list[InvalidToolCall]` — Tool calls with parsing errors
- **`usage_metadata`**: `UsageMetadata | None` — Token usage with optional breakdown fields

**Streaming variant**: `AIMessageChunk` — carries partial content and tool call chunks. Chunks merge intelligently: string content concatenates, tool call chunks with matching `index` merge their `args` strings, and usage counts sum.

### SystemMessage

Model instructions and priming.

```python
from langchain_core.messages import SystemMessage

msg = SystemMessage(
    content="You are a helpful assistant. Your name is Alice."
)
```

- **`type`**: Always `"system"`
- **Streaming variant**: `SystemMessageChunk`

### ToolMessage

Result of tool execution.

```python
from langchain_core.messages import ToolMessage

# Simple result
msg = ToolMessage(
    content="42",
    tool_call_id="call_abc123"
)

# With artifact (full output not sent to model)
msg = ToolMessage(
    content="The graph shows correlation of 0.85",
    artifact={"chart_data": [...]},
    tool_call_id="call_abc123",
    status="success"  # or "error"
)
```

**Unique fields**:
- **`tool_call_id`**: `str` — Associates this result with the AIMessage tool call
- **`artifact`**: `Any` — Optional data not sent to the model
- **`status`**: `Literal["success", "error"]` — Tool execution status

### ChatMessage

Arbitrary role message for custom participants.

```python
from langchain_core.messages import ChatMessage

msg = ChatMessage(role="Expert", content="Here's my analysis...")
```

- **`role`**: `str` — Custom speaker identifier
- **Streaming variant**: `ChatMessageChunk`

### FunctionMessage (Legacy)

Older variant of `ToolMessage` without `tool_call_id`. Prefer `ToolMessage`.

```python
from langchain_core.messages import FunctionMessage

msg = FunctionMessage(name="my_function", content="result")
```

### RemoveMessage

Message deletion marker for filtering.

```python
from langchain_core.messages import RemoveMessage

remove = RemoveMessage(id="msg_to_delete")
```

- **`type`**: Always `"remove"`
- Cannot contain `content`; raises `ValueError` if provided

## Serialization

### message_to_dict()

Convert a single message to a dictionary.

```python
from langchain_core.messages import message_to_dict, AIMessage

msg = AIMessage(content="Hello")
d = message_to_dict(msg)
# {"type": "ai", "data": {"content": "Hello", "additional_kwargs": {}, ...}}
```

**Parameters**:
- `message: BaseMessage` — Message to convert

**Returns**: `dict[str, Any]` with keys `"type"` (message type string) and `"data"` (message fields)

### messages_to_dict()

Convert a sequence of messages to a list of dicts.

```python
from langchain_core.messages import messages_to_dict, HumanMessage, AIMessage

messages = [
    HumanMessage(content="Hi"),
    AIMessage(content="Hello!")
]
dicts = messages_to_dict(messages)
```

**Parameters**:
- `messages: Sequence[BaseMessage]` — Messages to convert

**Returns**: `list[dict[str, Any]]` — List of message dicts

### messages_from_dict()

Reconstruct messages from dicts (reverse of `messages_to_dict()`).

```python
from langchain_core.messages import messages_from_dict

messages = messages_from_dict([
    {"type": "human", "data": {"content": "Hi"}},
    {"type": "ai", "data": {"content": "Hello!"}}
])
```

**Parameters**:
- `messages: Sequence[dict[str, Any]]` — Message dicts with `"type"` and `"data"` keys

**Returns**: `list[BaseMessage]` — Deserialized messages

**Uses internally**: `_message_from_dict()` for each message

### _message_from_dict()

Reconstruct a single message from a dict (helper for `messages_from_dict()`).

```python
from langchain_core.messages.utils import _message_from_dict

msg = _message_from_dict({
    "type": "human",
    "data": {"content": "Hi", "additional_kwargs": {}, ...}
})
```

**Parameters**:
- `v: dict[str, Any]` — Message dict with `"type"` and `"data"` keys

**Returns**: `BaseMessage` — Deserialized message

## Content Blocks

Content blocks represent standardized, provider-agnostic message content. Each block is a `TypedDict` with a `type` discriminator.

### TextContentBlock

Plain text content.

```python
from langchain_core.messages.content import TextContentBlock

block: TextContentBlock = {
    "type": "text",
    "text": "Hello, world!",
    # Optional fields
    "id": "text_1",
    "index": 0,
    "annotations": [],  # Citation/annotation metadata
}
```

### ImageContentBlock

Image content (URL or base64 data).

```python
from langchain_core.messages.content import ImageContentBlock

block: ImageContentBlock = {
    "type": "image",
    "url": "https://example.com/image.png",
    # or: "data": "iVBORw0KGgoAAAANS...", (base64)
    "mime_type": "image/png",  # Optional
}
```

### AudioContentBlock

Audio content.

```python
from langchain_core.messages.content import AudioContentBlock

block: AudioContentBlock = {
    "type": "audio",
    "url": "https://example.com/audio.wav",
    "mime_type": "audio/wav",  # Optional
}
```

### VideoContentBlock

Video content.

```python
from langchain_core.messages.content import VideoContentBlock

block: VideoContentBlock = {
    "type": "video",
    "url": "https://example.com/video.mp4",
    "mime_type": "video/mp4",  # Optional
}
```

### FileContentBlock

Generic file content.

```python
from langchain_core.messages.content import FileContentBlock

block: FileContentBlock = {
    "type": "file",
    "url": "https://example.com/document.pdf",
    "mime_type": "application/pdf",
}
```

### DataContentBlock

Arbitrary data (structured, binary, etc.).

```python
from langchain_core.messages.content import DataContentBlock

block: DataContentBlock = {
    "type": "data",
    "data": b"binary content",  # or dict, list, etc.
    "mime_type": "application/octet-stream",  # Optional
}
```

### ReasoningContentBlock

Reasoning/thinking content from models.

```python
from langchain_core.messages.content import ReasoningContentBlock

block: ReasoningContentBlock = {
    "type": "reasoning",
    "reasoning": "Let me think about this step by step...",
}
```

### NonStandardContentBlock

Provider-specific content not yet mapped to a standard block.

```python
from langchain_core.messages.content import NonStandardContentBlock

block: NonStandardContentBlock = {
    "type": "custom_provider_type",
    "data": {...},  # Provider-specific structure
}
```

## Tool Calling

### ToolCall

The model's request to execute a tool.

```python
from langchain_core.messages import ToolCall

call: ToolCall = {
    "type": "tool_call",
    "id": "call_abc123",
    "name": "web_search",
    "args": {"query": "Python"}  # Parsed dict, not JSON string
}
```

**Fields**:
- `type`: Always `"tool_call"`
- `id`: `str | None` — Unique identifier
- `name`: `str` — Tool name
- `args`: `dict[str, Any]` — Parsed arguments

### ToolCallChunk

A chunk of a tool call during streaming.

```python
from langchain_core.messages import ToolCallChunk

chunk: ToolCallChunk = {
    "type": "tool_call_chunk",
    "id": "call_abc123",
    "name": "search",
    "args": '{"query": "py',  # JSON string fragment
    "index": 0,  # Used for merging
}
```

**Fields**:
- `type`: Always `"tool_call_chunk"`
- `id`: `str | None`
- `name`: `str | None` — May be partial
- `args`: `str | None` — JSON string fragment
- `index`: `int | None` — For merging chunks with same index

**Merging**: When `AIMessageChunk` objects are concatenated (`+`), chunks with matching non-`None` index values have their `args` strings concatenated. Once complete, the `.tool_calls` property parses the accumulated JSON.

### InvalidToolCall

A tool call with parsing or validation errors.

```python
from langchain_core.messages import InvalidToolCall

invalid: InvalidToolCall = {
    "type": "invalid_tool_call",
    "id": "call_def456",
    "name": "search",
    "args": '{"query": "test"',  # Incomplete JSON
    "error": "Expecting '}'"
}
```

**Fields**:
- `type`: Always `"invalid_tool_call"`
- `id`: `str | None`
- `name`: `str` — Attempted tool name
- `args`: `str` — Raw argument string
- `error`: `str | None` — Error description

## Notification Metadata API

### NotificationMetadata

A `TypedDict` marking a message as system-generated by a specific feature.

```python
from langchain_core.messages import NotificationMetadata
from typing_extensions import TypedDict

class NotificationMetadata(TypedDict):
    """Metadata for messages the agent harness adds to a conversation."""
    source: str
    """The feature that added the message (e.g., 'async_subagent', 'resumable_checkpoint')."""
    
    data: NotRequired[dict[str, Any]]
    """Feature-specific fields, documented by that feature."""
```

**Purpose**: Marks human messages that the system added—not user-typed input—such as:
- Background task updates
- Conversation summaries
- Notifications from async subagents
- Resumable checkpoints or artifacts

**Availability**: Added in `langchain-core` 1.7.2+

**Storage**: Stored in `BaseMessage.additional_kwargs` as `lc_source` (and optionally `lc_data`). LangChain chat model integrations do not send these keys to the model.

### add_notification_metadata()

Mark a `HumanMessage` as system-generated.

```python
from langchain_core.messages import HumanMessage, add_notification_metadata

# Mark a message
message = HumanMessage("Background task t_42 finished.")
marked = add_notification_metadata(
    message,
    source="async_subagent",
    data={"task_id": "t_42", "status": "success"}
)

# marked.additional_kwargs["lc_source"] == "async_subagent"
# marked.additional_kwargs["lc_data"] == {"task_id": "t_42", "status": "success"}
```

**Signature**:

```python
def add_notification_metadata(
    message: HumanMessage,
    *,
    source: str,
    data: dict[str, Any] | None = None
) -> HumanMessage:
```

**Parameters**:
- `message: HumanMessage` — The message to mark
- `source: str` — The feature that added the message (e.g., `"async_subagent"`, `"conversation_summarizer"`)
- `data: dict[str, Any] | None` — Optional feature-specific data; must be JSON-serializable

**Returns**: `HumanMessage` — A shallow copy of the message with the same `id`, with `lc_source` and optional `lc_data` set in `additional_kwargs`. Overwrites any existing `lc_source` and `lc_data`.

**Behavior**:
- Only the `lc_source` and `lc_data` keys in `additional_kwargs` are modified; all other keys are preserved
- When `data` is `None`, only `lc_source` is set
- When two notified messages are merged (e.g., by `merge_message_runs()`), the metadata is preserved on the merged message and `lc_source` values are concatenated

**Security note**: Anyone who can send input to a thread can set these keys. Do not use them to decide whether to trust a message; they are informational only.

**Example**: Mark a background task notification

```python
from langchain_core.messages import HumanMessage, add_notification_metadata

# A notification from a background process
message = add_notification_metadata(
    HumanMessage("Task completed: Report generated and saved."),
    source="background_scheduler",
    data={
        "task_id": "report_gen_xyz",
        "completion_time": 125.4,
        "output_file": "s3://bucket/reports/report_xyz.pdf"
    }
)
```

### get_notification_metadata()

Retrieve notification metadata from a message.

```python
from langchain_core.messages import AIMessage, get_notification_metadata, add_notification_metadata

# Add metadata
msg = add_notification_metadata(
    HumanMessage("Status update"),
    source="monitoring",
    data={"level": "warning"}
)

# Retrieve it
metadata = get_notification_metadata(msg)
# metadata == {"source": "monitoring", "data": {"level": "warning"}}

# Non-notified message
msg2 = AIMessage("Hello")
metadata2 = get_notification_metadata(msg2)
# metadata2 is None
```

**Signature**:

```python
def get_notification_metadata(message: BaseMessage) -> NotificationMetadata | None:
```

**Parameters**:
- `message: BaseMessage` — Any message to check

**Returns**: `NotificationMetadata | None` — The metadata if `lc_source` is a string in `additional_kwargs`, otherwise `None`. The `data` field is omitted if `lc_data` is missing or not a dict (present only if `lc_data` is a dict).

**Behavior**:
- Returns `None` if `lc_source` is absent or not a string
- Returns a copy of `lc_data` (if present and a dict); the original message is not modified
- Safe to call on any `BaseMessage` type; returns `None` for non-notified messages

## Utility Functions

### filter_messages()

Filter messages based on name, type, ID, or tool calls.

```python
from langchain_core.messages import (
    filter_messages,
    HumanMessage, AIMessage, SystemMessage
)

messages = [
    SystemMessage("You are helpful."),
    HumanMessage("What is 2+2?", id="msg_1", name="user"),
    AIMessage("The answer is 4.", id="msg_2", name="assistant",
              tool_calls=[{"id": "call_1", "name": "calc", "args": {...}}]),
    HumanMessage("Thanks!", id="msg_3", name="user"),
]

# Keep only human and system messages
filtered = filter_messages(
    messages,
    include_types=["human", "system"]
)

# Include specific names, exclude a specific ID
filtered = filter_messages(
    messages,
    include_names=["user", "assistant"],
    exclude_ids=["msg_2"]
)

# Exclude tool calls (removes AIMessage and ToolMessage)
filtered = filter_messages(messages, exclude_tool_calls=True)

# Exclude specific tool call IDs (keeps AIMessage if other tool calls remain)
filtered = filter_messages(
    messages,
    exclude_tool_calls=["call_1"]
)
```

**Signature**:

```python
def filter_messages(
    messages: Iterable[MessageLikeRepresentation] | PromptValue,
    *,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    include_types: Sequence[str | type[BaseMessage]] | None = None,
    exclude_types: Sequence[str | type[BaseMessage]] | None = None,
    include_ids: Sequence[str] | None = None,
    exclude_ids: Sequence[str] | None = None,
    exclude_tool_calls: Sequence[str] | bool | None = None,
) -> list[BaseMessage]:
```

**Parameters**:
- `include_names`, `exclude_names`: Filter by `name` field
- `include_types`, `exclude_types`: Filter by type (string or class); e.g., `"ai"` or `AIMessage`
- `include_ids`, `exclude_ids`: Filter by `id` field
- `exclude_tool_calls`:
  - `True`: Remove all AIMessages with tool calls and all ToolMessages
  - `Sequence[str]`: Remove ToolMessages with these tool call IDs; update AIMessages to exclude matching tool calls; remove AIMessages if no tool calls remain

**Returns**: `list[BaseMessage]` — Messages matching at least one `include_*` condition and none of the `exclude_*` conditions

### merge_message_runs()

Merge consecutive messages of the same type.

```python
from langchain_core.messages import merge_message_runs, HumanMessage, AIMessage

messages = [
    HumanMessage("What's your favorite color?", id="msg_1"),
    HumanMessage("Actually, favorite food?", id="msg_2"),
    AIMessage("Blue", id="msg_3"),
    AIMessage("Lasagna", id="msg_4"),
]

merged = merge_message_runs(messages)
# Result:
# [
#   HumanMessage("What's your favorite color?\nActually, favorite food?", id="msg_1"),
#   AIMessage("Blue\nLasagna", id="msg_3"),
# ]
```

**Signature**:

```python
def merge_message_runs(
    messages: Iterable[MessageLikeRepresentation] | PromptValue,
    *,
    chunk_separator: str = "\n",
) -> list[BaseMessage]:
```

**Parameters**:
- `messages` — Sequence of messages to merge
- `chunk_separator: str` — String to insert between merged text content; defaults to `"\n"`

**Returns**: `list[BaseMessage]` — Consecutive messages of the same type merged into single messages

**Behavior**:
- If both messages have string content, merged content is a string with `chunk_separator` between them
- If at least one message has list content, merged content is a list of content blocks
- ToolMessages are never merged (each has a distinct `tool_call_id`)
- Tool calls, usage metadata, and other fields are combined intelligently
- Notification metadata on merged messages is preserved and source strings are concatenated

### message_chunk_to_message()

Convert a `MessageChunk` to its non-chunk variant.

```python
from langchain_core.messages import (
    message_chunk_to_message,
    AIMessageChunk
)

chunk = AIMessageChunk(content="Hello", id="msg_1")
msg = message_chunk_to_message(chunk)
# Returns AIMessage(content="Hello", id="msg_1")

# If input is already a non-chunk message, returns it unchanged
from langchain_core.messages import AIMessage
msg2 = AIMessage(content="Hello")
result = message_chunk_to_message(msg2)
# result is msg2
```

**Signature**:

```python
def message_chunk_to_message(chunk: BaseMessage) -> BaseMessage:
```

**Parameters**:
- `chunk: BaseMessage` — A message chunk or regular message

**Returns**: `BaseMessage` — The non-chunk variant; if input is already non-chunk, returns it unchanged

### trim_messages()

Trim a message list to fit within token/character budget.

```python
from langchain_core.messages import trim_messages, HumanMessage, AIMessage

messages = [
    HumanMessage("First question"),
    AIMessage("First answer " * 100),
    HumanMessage("Second question"),
    AIMessage("Second answer"),
]

# Keep most recent, trim to ~100 tokens
trimmed = trim_messages(
    messages,
    max_tokens=100,
    strategy="last",  # Keep most recent messages
    token_counter=tiktoken_counter,  # or other counter
)
```

**Signature**:

```python
def trim_messages(
    messages: Sequence[BaseMessage],
    max_tokens: int,
    *,
    strategy: Literal["last", "first", "sliding_window"] = "last",
    token_counter: Callable[[str], int] | Callable[[BaseMessage], int] | None = None,
    end_on: Sequence[str | type[BaseMessage]] | None = None,
    start_on: Sequence[str | type[BaseMessage]] | None = None,
) -> list[BaseMessage]:
```

**Parameters**:
- `messages` — Sequence to trim
- `max_tokens: int` — Maximum total tokens
- `strategy`: `"last"` (keep most recent), `"first"` (keep oldest), or `"sliding_window"` (keep recent + some old)
- `token_counter` — Function to count tokens in a message or string
- `end_on`, `start_on` — Message types to use as boundaries

**Returns**: `list[BaseMessage]` — Trimmed messages

### get_buffer_string()

Render messages as a formatted string.

```python
from langchain_core.messages import get_buffer_string, HumanMessage, AIMessage

messages = [
    HumanMessage("Hello"),
    AIMessage("Hi there!"),
]

buffer = get_buffer_string(messages)
# Output:
# "Human: Hello\nAI: Hi there!"
```

**Signature**:

```python
def get_buffer_string(
    messages: Sequence[BaseMessage],
    *,
    human_prefix: str = "Human",
    ai_prefix: str = "AI",
) -> str:
```

**Parameters**:
- `messages` — Sequence of messages
- `human_prefix: str` — Prefix for HumanMessage (default: `"Human"`)
- `ai_prefix: str` — Prefix for AIMessage (default: `"AI"`)

**Returns**: `str` — Formatted message buffer

## Use Cases: Notification Metadata

Notification metadata enables tracking of system-generated messages without affecting model communication. Common use cases:

### Background Task Updates

```python
from langchain_core.messages import HumanMessage, add_notification_metadata

# A background process generates a status update
async def background_task():
    # ... do work ...
    update = HumanMessage("Background job 'data_sync' completed successfully.")
    return add_notification_metadata(
        update,
        source="background_scheduler",
        data={"job_id": "data_sync", "duration_seconds": 45.2}
    )
```

### Conversation Summarization

```python
from langchain_core.messages import HumanMessage, add_notification_metadata

def create_summary_notification(summary_text: str):
    """Create a system-generated summary of earlier conversation."""
    return add_notification_metadata(
        HumanMessage(summary_text),
        source="summarizer",
        data={"type": "auto_summary", "window_messages": 10}
    )
```

### Async Subagent Results

```python
from langchain_core.messages import HumanMessage, add_notification_metadata

def async_subagent_result(result: str, subagent_name: str):
    """Mark a result from an async subagent."""
    return add_notification_metadata(
        HumanMessage(f"Subagent '{subagent_name}' result: {result}"),
        source="async_subagent",
        data={"subagent": subagent_name, "completion_time": 12.5}
    )
```

### Workflow Checkpoints

```python
from langchain_core.messages import HumanMessage, add_notification_metadata
import json

def resumable_checkpoint(state_snapshot: dict):
    """Record a resumable checkpoint."""
    return add_notification_metadata(
        HumanMessage("Checkpoint saved."),
        source="resumable_checkpoint",
        data={"state_id": "chk_xyz", "timestamp": 1729000000}
    )
```

### Message Filtering with Metadata

When you need to inspect which messages were user-typed vs. system-generated:

```python
from langchain_core.messages import get_notification_metadata

def get_user_messages(messages):
    """Extract only user-typed messages, excluding system-generated ones."""
    user_messages = []
    for msg in messages:
        # Skip system-generated messages
        if msg.type == "human" and get_notification_metadata(msg) is None:
            user_messages.append(msg)
    return user_messages

def get_system_notifications(messages):
    """Extract all system-generated messages with their metadata."""
    notifications = []
    for msg in messages:
        metadata = get_notification_metadata(msg)
        if metadata is not None:
            notifications.append((msg, metadata))
    return notifications
```

## Serialization Workflow Example

Complete example: create, serialize, deserialize, and add metadata to messages.

```python
from langchain_core.messages import (
    HumanMessage, AIMessage, message_to_dict, messages_to_dict,
    messages_from_dict, add_notification_metadata, get_notification_metadata
)
import json

# 1. Create messages
user_msg = HumanMessage("What is LLM?")
ai_msg = AIMessage("An LLM (Large Language Model) is...")

messages = [user_msg, ai_msg]

# 2. Serialize to dict
dicts = messages_to_dict(messages)
print(json.dumps(dicts, indent=2))
# Output:
# [
#   {
#     "type": "human",
#     "data": {
#       "content": "What is LLM?",
#       "additional_kwargs": {},
#       ...
#     }
#   },
#   {
#     "type": "ai",
#     "data": {
#       "content": "An LLM (Large Language Model) is...",
#       ...
#     }
#   }
# ]

# 3. Serialize to JSON string and store
json_str = json.dumps(dicts)
stored_messages = json_str

# 4. Deserialize from JSON string
recovered_dicts = json.loads(stored_messages)
recovered_messages = messages_from_dict(recovered_dicts)

# 5. Add notification to one of the messages
notified_msg = add_notification_metadata(
    recovered_messages[0],
    source="storage_service",
    data={"loaded_from": "s3://bucket/messages", "timestamp": 1729000000}
)

# 6. Check metadata
metadata = get_notification_metadata(notified_msg)
print(metadata)
# Output: {"source": "storage_service", "data": {"loaded_from": "...", ...}}

# 7. Re-serialize with notification
updated_dicts = messages_to_dict([notified_msg, recovered_messages[1]])
print(updated_dicts[0]["data"]["additional_kwargs"])
# Output: {"lc_source": "storage_service", "lc_data": {"loaded_from": "...", ...}}
```

## Best Practices

1. **Serialization for persistence**: Use `messages_to_dict()` before storing, and `messages_from_dict()` to restore. This ensures all message types and metadata are preserved.

2. **Notification metadata only for HumanMessage**: Only call `add_notification_metadata()` on `HumanMessage` instances. It is designed to mark user-like input that was actually generated by the system.

3. **Content blocks over raw dicts**: Prefer `.content_blocks` property for accessing structured content instead of manually parsing raw content lists.

4. **Message filtering before inference**: Use `filter_messages()` to remove internal metadata (tool calls, artifacts) before sending to downstream models.

5. **Merging for cleanup**: Use `merge_message_runs()` to consolidate consecutive messages of the same type before serializing or passing to models.

6. **Token budget management**: Use `trim_messages()` with an appropriate token counter (e.g., from `tiktoken`) to keep message histories within model context limits.

7. **Metadata for auditability**: Use notification metadata to mark and track system-generated input, but do not rely on it for security decisions.

8. **Chunk accumulation**: When streaming, accumulate `MessageChunk` objects (using `+`) until streaming is complete, then convert to non-chunk messages via `message_chunk_to_message()`.
