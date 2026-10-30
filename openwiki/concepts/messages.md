---
type: "Concept"
title: "Messages: Core Abstraction for LLM Conversations"
description: "Core message types (AIMessage, HumanMessage, ToolMessage, SystemMessage), content blocks, serialization, filtering utilities, notification metadata, and the message protocol at the heart of LangChain workflows."
tags: ["messages", "content-blocks", "tool-calling", "multimodal", "streaming", "message-chunks", "notification-metadata"]
sources:
  - id: openwiki-source-0f6ea1dd09fb4675ff4112b1
    resource: repo://libs/core/langchain_core/messages/__init__.py
  - id: openwiki-source-77dc1fb726463969f9d53658
    resource: repo://libs/core/langchain_core/messages/ai.py
  - id: openwiki-source-b32b84365d17276620c41ebc
    resource: repo://libs/core/langchain_core/messages/base.py
  - id: openwiki-source-2e77747f30fe980d17d5d1a2
    resource: repo://libs/core/langchain_core/messages/block_translators/__init__.py
  - id: openwiki-source-ac2e0f8b0fb1cb3b223672b7
    resource: repo://libs/core/langchain_core/messages/block_translators/openai.py
  - id: openwiki-source-ecd2d8742bfabb625fcb2773
    resource: repo://libs/core/langchain_core/messages/chat.py
  - id: openwiki-source-fc874ddb29b9c5840565397f
    resource: repo://libs/core/langchain_core/messages/content.py
  - id: openwiki-source-08aa8259bef677b0e8686d97
    resource: repo://libs/core/langchain_core/messages/function.py
  - id: openwiki-source-8bb392f5dbc1fe7faaf52430
    resource: repo://libs/core/langchain_core/messages/human.py
  - id: openwiki-source-426ea59384b70ca69477f42a
    resource: repo://libs/core/langchain_core/messages/modifier.py
  - id: openwiki-source-56c6ea6a11f04b5c0e01aa7b
    resource: repo://libs/core/langchain_core/messages/notification.py
  - id: openwiki-source-dad8cfeb38a829e03e165986
    resource: repo://libs/core/langchain_core/messages/system.py
  - id: openwiki-source-9861ba5cf0c42c142cf732f9
    resource: repo://libs/core/langchain_core/messages/tool.py
  - id: openwiki-source-498a9586e021b126ab8a8b42
    resource: repo://libs/core/langchain_core/messages/utils.py
  - id: openwiki-source-9543ce826699c4339c2915af
    resource: repo://libs/core/langchain_core/utils/_merge.py
  - id: openwiki-source-6f7b6a330c61bc2d330a1457
    resource: repo://libs/core/tests/unit_tests/messages/test_notification.py
  - id: openwiki-source-5b54a79fff5b1358acd7d4f3
    resource: repo://libs/core/tests/unit_tests/test_messages.py
generated: { by: "openwiki/0.5.0", at: "2026-10-30T08:19:17.669Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-30T08:19:17.669Z
---

# Messages: Core Abstraction for LLM Conversations

Messages are the **data contract** between application code and language models. They form the I/O abstraction for every conversation: human input flows in as `HumanMessage`, the model responds with `AIMessage`, tools execute and return results via `ToolMessage`, and system instructions prime model behavior via `SystemMessage`. This document covers the message class hierarchy, message content representations (including multimodal content blocks), tool-related types, streaming and message chunk concatenation semantics, notification metadata for marking system-generated messages, and provider-specific translation patterns.

## Core Message Hierarchy

All messages inherit from **`BaseMessage`**, a serializable object with a unified interface for storing content and metadata. The hierarchy supports both singular messages and streaming message chunks (which can be concatenated during streaming).

### BaseMessage: Common Foundation

**`BaseMessage`** (in `langchain_core.messages.base`) defines the shared contract:

- **`content`**: `str | list[str | dict[Any, Any]]` — The actual message payload. Can be:
  - A plain string (e.g., `"What is 2+2?"`)
  - A list of content blocks—strings, dicts representing multimodal content (text, images, audio, etc.), or tool calls
  
- **`type`**: `str` — A unique, serializable discriminator (e.g., `"human"`, `"ai"`, `"system"`, `"tool"`). Used for deserialization.

- **`id`**: `str | None` — Optional unique identifier, typically assigned by the provider or LangChain's run manager. Enables association with traces and logs.

- **`name`**: `str | None` — Optional speaker/function name. Used by `FunctionMessage` and `ChatMessage` to indicate role or function identity.

- **`additional_kwargs`**: `dict[Any, Any]` — Reserved field for provider-specific data (e.g., OpenAI's full `tool_calls` array, function call metadata, refusal flags). Not used by core LangChain code but preserved for downstream integration.

- **`response_metadata`**: `dict[Any, Any]` — Response headers, token usage, model name, finish reason, logprobs, and other provider metadata extracted from API responses.

**Key behavior**:
- **Serialization**: `BaseMessage.is_lc_serializable()` returns `True`, enabling roundtrip via `message_to_dict()` and `messages_from_dict()`.
- **Text property**: `.text` property extracts text content from both string and list-based content, ignoring non-text blocks; supports both property access (`.text`) and legacy method access (`.text()`).
- **Content blocks**: `.content_blocks` property parses message content into standard, typed `ContentBlock` objects, handling provider-specific format translation when needed.
- **String concatenation**: `message + other_message` creates a `ChatPromptTemplate` for composition with other prompts.

### Standard Message Types

#### HumanMessage
**User input to the model.** Represents the human's turn in a conversation.

```python
from langchain_core.messages import HumanMessage

msg = HumanMessage(content="What is the capital of France?")
msg = HumanMessage(content=[
    {"type": "text", "text": "Describe this image"},
    {"type": "image", "url": "https://example.com/image.png"}
])
```

- **`type`**: Always `"human"` (discriminator for serialization).
- **Streaming variant**: `HumanMessageChunk` — used internally during streaming; can be concatenated with `+` operator.

#### AIMessage
**Model response.** Contains the model's output, potentially including tool calls and usage metadata.

```python
from langchain_core.messages import AIMessage

# Plain text response
msg = AIMessage(content="The capital of France is Paris.")

# With tool calls
msg = AIMessage(
    content="I'll search for that information.",
    tool_calls=[
        {"type": "tool_call", "id": "call_123", "name": "search", "args": {"query": "..."}}
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
- **`tool_calls`**: `list[ToolCall]` — If present, tool invocations requested by the model. Each `ToolCall` is a dict with `name`, `args` (parsed JSON dict), `id`, and optional metadata.
- **`invalid_tool_calls`**: `list[InvalidToolCall]` — Tool calls with parsing errors (malformed JSON arguments, etc.). Includes an `error` field describing the issue.
- **`usage_metadata`**: `UsageMetadata | None` — Standardized token usage: `input_tokens`, `output_tokens`, `total_tokens`, and optional breakdown fields (`input_token_details`, `output_token_details`) for cache hits/misses, audio tokens, reasoning tokens, etc.

**Streaming variant**: `AIMessageChunk` — carries partial content, tool call chunks, and usage metadata. Chunks merge intelligently when added:
- String content concatenates.
- Tool call chunks with matching `index` fields merge their `args` strings.
- Usage counts sum across chunks.
- ID precedence: provider-assigned IDs (`msg_...`) take precedence over LangChain-assigned run IDs (`lc_run_...`).

#### SystemMessage
**Model instructions/priming.** Establishes the model's behavior and persona.

```python
from langchain_core.messages import SystemMessage

msg = SystemMessage(
    content="You are a helpful assistant. Your name is Alice. Respond concisely."
)
```

- **`type`**: Always `"system"`.
- **Streaming variant**: `SystemMessageChunk`.

#### ToolMessage
**Result of tool execution.** Completes a tool call initiated by the model.

```python
from langchain_core.messages import ToolMessage

# Simple result
msg = ToolMessage(
    content="42",
    tool_call_id="call_Jja7J89XsjrOLA5r"
)

# With artifact (full output not sent to model)
msg = ToolMessage(
    content="The graph shows correlation of 0.85",
    artifact={"full_output": "...", "chart_data": [...] },
    tool_call_id="call_123",
    status="success"  # or "error"
)
```

**Unique fields**:
- **`tool_call_id`**: `str` — ID of the `AIMessage` tool call this result responds to. Used to associate parallel tool executions with their results.
- **`artifact`**: `Any` — Optional data not sent to the model (e.g., full output when message carries only a summary).
- **`status`**: `Literal["success", "error"]` — Tool execution status; defaults to `"success"`.

**Design note**: `ToolMessage` objects returned from tool execution may implement `ToolOutputMixin` to signal they can be directly invoked; non-mixin outputs are coerced to strings and wrapped in a `ToolMessage`.

#### FunctionMessage (Legacy)
**Older version of `ToolMessage` without `tool_call_id`.** Used for OpenAI-specific function calling in legacy code; `ToolMessage` is preferred.

```python
from langchain_core.messages import FunctionMessage

msg = FunctionMessage(name="my_function", content="result")
```

#### ChatMessage
**Arbitrary role message.** Allows custom speaker roles beyond the standard human/AI/system/tool set.

```python
from langchain_core.messages import ChatMessage

msg = ChatMessage(role="Expert", content="Here's my analysis...")
```

- **`role`**: `str` — Custom speaker identifier (not limited to predefined values).
- Useful for multi-turn conversations with specialized participants.

#### RemoveMessage
**Message deletion marker.** Used in message filtering to remove a message from history by ID.

```python
from langchain_core.messages import RemoveMessage

remove = RemoveMessage(id="msg_to_delete")
```

- **`type`**: Always `"remove"`.
- **Constraint**: Does not support `content` field; raises `ValueError` if provided.

## Tool-Related Message Types and Tool Calls

### ToolCall: The Tool Invocation Request

A **`ToolCall`** is a `TypedDict` representing the model's request to execute a tool:

```python
from langchain_core.messages import ToolCall

tool_call = {
    "type": "tool_call",
    "id": "call_abc123",           # Unique ID for this call
    "name": "web_search",          # Tool name
    "args": {"query": "Python"}    # Parsed arguments (dict, not JSON string)
}
```

**Fields**:
- **`name`**: `str` — Name of the tool to invoke (matches the tool's schema name).
- **`args`**: `dict[str, Any]` — Parsed argument dictionary; the model outputs JSON, but `ToolCall` always contains the parsed dict.
- **`id`**: `str | None` — Unique identifier, essential when the model makes multiple concurrent tool calls.
- **`type`**: `"tool_call"` — Discriminator.

**Factory function**: `tool_call(name=..., args=..., id=...)` validates and creates a `ToolCall` with strict type checking.

### ToolCallChunk: Streaming Tool Calls

During streaming, the model may emit tool call arguments incrementally as JSON fragments. **`ToolCallChunk`** represents a chunk:

```python
from langchain_core.messages import ToolCallChunk

chunk = {
    "type": "tool_call_chunk",
    "id": "call_abc123",        # Same ID as final ToolCall
    "name": "search",           # May be partial initially
    "args": '{"query": "py',    # JSON string fragment, not parsed
    "index": 0,                 # Index for merging
}
```

**Fields**:
- **`name`**: `str | None` — Tool name; may be `None` in middle chunks.
- **`args`**: `str | None` — JSON string fragment; accumulated and parsed when complete.
- **`id`**: `str | None` — Tool call ID.
- **`index`**: `int | None` — Used during merging: chunks with the same non-`None` index merge their string fields, enabling proper JSON accumulation.

**Merging semantic**: When `AIMessageChunk` objects are concatenated via `+`, tool call chunks with matching `index` values have their `args` strings concatenated. Once complete, `.tool_calls` property parses the JSON strings into final `ToolCall` objects.

### InvalidToolCall: Parsing Errors

When the model generates invalid JSON arguments or violates the tool schema, a **`InvalidToolCall`** captures the error:

```python
invalid = {
    "type": "invalid_tool_call",
    "id": "call_def456",
    "name": "search",
    "args": '{"query": "test"',   # Incomplete JSON
    "error": "Expecting '}' after 'test'"
}
```

**Fields**:
- **`name`**, **`args`**, **`id`**: Same as `ToolCall`.
- **`error`**: `str | None` — Error message describing why the tool call is invalid.

**Factory function**: `invalid_tool_call(name=..., args=..., id=..., error=...)` — validates at creation time.

### Tool Call Flow Example

```
User: "Search for Python tutorials"
  ↓
HumanMessage(content="Search for Python tutorials")
  ↓
[Model invoked with bound search tool]
  ↓
AIMessage(
    content="I'll search for that.",
    tool_calls=[
        {"id": "call_1", "name": "search", "args": {"query": "Python tutorials"}}
    ]
)
  ↓
[Tool executor finds and runs search tool]
  ↓
ToolMessage(
    tool_call_id="call_1",
    content="Found: [3 Python tutorial links]"
)
  ↓
[Model invokes again with tool result in history]
  ↓
AIMessage(content="Here are 3 Python tutorials...")
```

## Content Blocks: Multimodal Message Structure

Messages support **content blocks**—a unified, provider-agnostic representation of multimodal content. A message's `content` field can be a list of blocks; each block is a `TypedDict` with a `type` field.

### Standard Content Block Types

#### TextContentBlock
Plain text content from the model or user.

```python
{
    "type": "text",
    "text": "Hello, world!",
    "id": "lc_abc123",                  # Optional unique ID
    "annotations": [...],               # Optional citations/references
    "index": 0,                         # Used during streaming
    "extras": {"custom_field": "..."}   # Provider-specific metadata
}
```

#### ImageContentBlock
Image data (URL or base64).

```python
{
    "type": "image",
    "url": "https://example.com/image.png",
    # OR
    "base64": "iVBORw0KGgoAAAAN...",
    "mime_type": "image/png",  # Required for base64
    "file_id": "file_123",     # Reference to external storage (OpenAI Files API)
}
```

#### AudioContentBlock
Audio data (URL or base64).

```python
{
    "type": "audio",
    "base64": "...",
    "mime_type": "audio/wav"
}
```

#### VideoContentBlock
Video data (URL or base64).

```python
{
    "type": "video",
    "url": "https://example.com/video.mp4",
    "mime_type": "video/mp4"
}
```

#### FileContentBlock
Generic file data (PDFs, Word docs, etc.).

```python
{
    "type": "file",
    "base64": "...",
    "mime_type": "application/pdf",
    "filename": "document.pdf"
}
```

#### PlainTextContentBlock
Plain text files (`.txt`, `.md`).

```python
{
    "type": "text-plain",
    "text": "# My Document\nContent here...",
    "mime_type": "text/plain",
    "title": "My Document"
}
```

#### ToolCall Block
Tool call represented within content (used when constructing AIMessage with content_blocks).

```python
{
    "type": "tool_call",
    "id": "call_123",
    "name": "search",
    "args": {"query": "..."}
}
```

#### ReasoningContentBlock
Model's reasoning or thinking process (from models like o1, deepseek, etc.).

```python
{
    "type": "reasoning",
    "reasoning": "Let me break down this problem step by step..."
}
```

#### ServerToolCall / ServerToolResult
Server-side tool execution blocks (for models that execute tools themselves).

```python
{
    "type": "server_tool_call",
    "id": "server_123",
    "name": "code_execution",
    "args": {"code": "..."}
}

{
    "type": "server_tool_result",
    "tool_call_id": "server_123",
    "status": "success",
    "output": "..."
}
```

#### NonStandardContentBlock
Provider-specific content that doesn't map to standard blocks.

```python
{
    "type": "non_standard",
    "value": {"provider_specific_field": "..."}
}
```

### Content Block Parsing

**Key properties**:
- **`BaseMessage.content_blocks`**: Returns a list of typed `ContentBlock` objects by parsing the raw `content` field. Uses provider-specific translators (OpenAI, Anthropic, etc.) when available; falls back to best-effort parsing.
- **`BaseMessage.text`**: Extracts and concatenates all text from text-type blocks, ignoring other types.

**Provider translation workflow**:
1. If `response_metadata.get("model_provider")` is set (e.g., `"openai"`, `"anthropic"`), call the registered translator for that provider.
2. Otherwise, use best-effort parsing: split content into blocks, translate known block types, wrap unknown as `non_standard`.
3. For `AIMessage`, additionally extract tool calls from the `tool_calls` field into `ToolCall` blocks and extract reasoning from `additional_kwargs`.

## Message Metadata and Lifecycle

### ID Assignment and Tracking

Message IDs enable tracing and deduplication:

- **User-provided IDs**: Preferred when provided (e.g., `"msg_abc123"`).
- **LangChain-assigned IDs**: Generated by the run manager as `f"lc_run_{uuid4()}"` during execution. Used for internal tracking.
- **Provider IDs**: Extracted from API responses (e.g., OpenAI's message IDs).

**ID precedence** (especially in streaming):
1. Provider-assigned IDs (highest priority).
2. User-provided IDs.
3. LangChain-assigned run IDs.
4. `None` (lowest priority).

### Token Usage and Response Metadata

**`usage_metadata`** (on `AIMessage`):
- **`input_tokens`**: Total input token count.
- **`output_tokens`**: Total output token count.
- **`total_tokens`**: Sum of input and output.
- **`input_token_details`** (optional): Breakdown with keys like `audio`, `cache_creation`, `cache_read`.
- **`output_token_details`** (optional): Breakdown with keys like `audio`, `reasoning`.

**`response_metadata`**:
- **`finish_reason`**: Why generation stopped (e.g., `"stop"`, `"tool_calls"`, `"length"`).
- **`logprobs`**: Per-token log probabilities (when requested).
- **`model_name`**: Name of the model used.
- **`model_provider`**: Provider identifier (e.g., `"openai"`, `"anthropic"`).
- Other provider-specific fields (request/response headers, etc.).

**Example**:
```python
msg = AIMessage(
    content="...",
    usage_metadata={
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "input_token_details": {"cache_read": 20},
    },
    response_metadata={
        "finish_reason": "stop",
        "model_name": "gpt-4",
        "model_provider": "openai"
    }
)
```

### Notification Metadata: Marking System-Generated Messages

LangChain provides **notification metadata** utilities to mark `HumanMessage` objects that are generated by the system (not by the user), such as background task updates, conversation summaries, or notifications from async subagents. These utilities preserve metadata during message merging and filtering, ensuring system-generated messages are correctly tracked.

**Key functions**:

- **`add_notification_metadata(message, source=..., data=...)`**: Returns a copy of a `HumanMessage` marked with `additional_kwargs["lc_source"]` and optional `additional_kwargs["lc_data"]`. LangChain chat model integrations do not send these keys to the model.

- **`get_notification_metadata(message)`**: Returns a `NotificationMetadata` dict with `source` and optional `data` fields, or `None` if the message lacks `lc_source`.

**NotificationMetadata** (TypedDict):
- **`source`**: `str` — The feature that added the message (e.g., `"async_subagent"`, `"task_monitor"`, `"summary_generator"`).
- **`data`**: `dict[str, Any]` (optional) — Feature-specific fields, documented by that feature. Must be JSON-serializable.

**Example: Background task update**:
```python
from langchain_core.messages import HumanMessage, add_notification_metadata

# Mark a system-generated message
message = add_notification_metadata(
    HumanMessage("Background task t_42 completed successfully."),
    source="async_subagent",
    data={"task_id": "t_42", "status": "success", "result": "..."}
)

# Later, read the metadata
from langchain_core.messages import get_notification_metadata

metadata = get_notification_metadata(message)
if metadata:
    print(f"Source: {metadata['source']}")  # "async_subagent"
    if "data" in metadata:
        print(f"Task ID: {metadata['data']['task_id']}")  # "t_42"
```

**Example: Conversation summary**:
```python
from langchain_core.messages import add_notification_metadata

# Add a summary message marked as system-generated
summary = add_notification_metadata(
    HumanMessage("Summary so far: User asked about Python, received 3 tutorials."),
    source="conversation_summarizer",
    data={"summary_type": "interim", "message_count": 5}
)
```

**Preservation during merging and filtering**:

- **`merge_message_runs()`**: When merging consecutive `HumanMessage` objects, the merged message retains the first message's notification metadata and concatenates `lc_source` fields from multiple messages.
- **`filter_messages()`**: Notification metadata is preserved in `additional_kwargs` during filtering; the metadata remains intact on filtered messages.
- **Message chunk concatenation**: During streaming, `additional_kwargs` (including notification metadata) is recursively merged via the `+` operator.

**Important notes**:
- Notification metadata is stored in `additional_kwargs` and is **not** sent to the language model by LangChain's integrations.
- The metadata is JSON-serializable and survives message serialization/deserialization.
- Anyone who can send input to a thread can set these keys; do not use `lc_source`/`lc_data` alone to decide whether to trust a message.
- This feature was added in `langchain-core` 1.7.2.

## Message Serialization and Deserialization

Messages are fully serializable for storage and RPC:

```python
from langchain_core.messages import message_to_dict, messages_from_dict

# Serialize
msg_dict = message_to_dict(HumanMessage(content="Hello"))

# Deserialize
msg = messages_from_dict([msg_dict])[0]
```

**Format**: Each message becomes a dict with:
- **`type`**: Discriminator for deserialization.
- **`data`**: Nested dict with all fields (content, name, id, additional_kwargs, response_metadata, etc.).
- **`lc`** (optional): LangChain metadata (namespace, package version, etc.).

## Provider-Specific Message Conversion

Different LLM providers use different message formats. LangChain abstracts these via **block translators**.

### OpenAI Conversion Pattern

OpenAI expects messages with `role` (`"user"`, `"assistant"`, `"system"`) and `content` (string or list of content objects with `type` and nested data).

**LangChain → OpenAI**:
- `HumanMessage` → `{"role": "user", ...}`
- `AIMessage` → `{"role": "assistant", ...}`
- `SystemMessage` → `{"role": "system", ...}`
- `ToolMessage` → `{"role": "tool", "tool_call_id": "...", ...}`

**Content block translation** (via `convert_to_openai_data_block`):
- `ImageContentBlock` with URL → `{"type": "image_url", "image_url": {"url": "..."}}`
- `ImageContentBlock` with base64 → `{"type": "image_url", "image_url": {"url": "data:mime_type;base64,..."}}`
- `AudioContentBlock` → `{"type": "input_audio", "input_audio": {"data": "...", "format": "..."}}`
- `FileContentBlock` → `{"type": "file", "file": {"file_id": "..." or "file_data": "data:..."}}`

**Tool call handling**:
- `AIMessage.tool_calls` → Encoded in `additional_kwargs` or as part of response structure.
- Parsing occurs in reverse: extract tool calls from API response and populate `AIMessage.tool_calls` field.

### Anthropic Conversion Pattern

Anthropic expects a similar role/content structure but uses different content types and tool call IDs with stricter requirements.

**Content block translation differences**:
- `ImageContentBlock` → `{"type": "image", "source": {"type": "base64" or "url", ...}}`
- Tool call IDs must be non-numeric strings; LangChain sanitizes if needed.

### Provider Translator Registration

External providers (outside langchain-core) can register custom translators:

```python
from langchain_core.messages.block_translators import register_translator

def translate_my_provider(message: AIMessage) -> list[ContentBlock]:
    # Custom parsing logic
    return [...]

register_translator("my_provider", translate_my_provider, translate_my_provider_chunk)
```

## Streaming and MessageChunk

Streaming operations yield **message chunks** (e.g., `AIMessageChunk`, `HumanMessageChunk`, `ToolMessageChunk`). These are optimized for incremental accumulation via the `+` operator. All message chunk classes inherit from `BaseMessageChunk` and support concatenation; `AIMessageChunk` has specialized merging logic for tool calls and usage metadata.

### MessageChunk Concatenation: BaseMessageChunk

The base `BaseMessageChunk.__add__` operator merges:

```python
chunk1 = HumanMessageChunk(content="Hello", id="msg_1")
chunk2 = HumanMessageChunk(content=" World")

result = chunk1 + chunk2
# → HumanMessageChunk(content="Hello World", id="msg_1")
```

**Base merging rules**:
- **String content**: Concatenated directly.
- **List content**: Intelligently merged:
  - Dicts with matching `index` field merge their fields (via `merge_lists`).
  - Dicts without `index` or with non-matching `index` are appended.
- **additional_kwargs** and **response_metadata**: Recursively merged (nested dicts merged, lists concatenated).
- **ID precedence**: Provider-assigned IDs (non-`lc_*`, non-`lc_run_*`) > LangChain-assigned run IDs (`lc_run_*` prefix) > other IDs.

### AIMessageChunk Concatenation: Tool Calls and Usage

`AIMessageChunk` extends the base concatenation with specialized merging for streaming tool calls and token usage:

```python
chunks = [
    AIMessageChunk(content="", tool_call_chunks=[
        {"index": 0, "id": "call_1", "name": "search", "args": '{"query":', ...}
    ]),
    AIMessageChunk(content="", tool_call_chunks=[
        {"index": 0, "id": None, "name": None, "args": ' "Python"}', ...}
    ]),
]

final = chunks[0] + chunks[1]
# final.tool_call_chunks[0]["args"] == '{"query": "Python"}'
```

**AIMessageChunk-specific merging**:
- **tool_call_chunks**: Chunks with matching `index` have their `args` strings concatenated, enabling JSON argument accumulation during streaming.
- **usage_metadata**: Token counts (`input_tokens`, `output_tokens`, `total_tokens`) summed across chunks. Detailed breakdowns (`input_token_details`, `output_token_details`) recursively merged.
- **chunk_position**: Set to `"last"` if any chunk has `chunk_position="last"`; signals that tool_call_chunks should be parsed into tool_calls upon aggregation.

**ID precedence** (same as base, applied consistently across chunks):
1. Provider-assigned IDs (highest priority).
2. LangChain-assigned run IDs.
3. User-provided IDs.
4. `None` (lowest priority).

**Complete example with tool calls and usage**:
```python
from langchain_core.messages import AIMessageChunk

chunks = [
    AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"index": 0, "id": "call_1", "name": "search", "args": '{"query":', ...}
        ],
        usage_metadata={"input_tokens": 100, "output_tokens": 0, "total_tokens": 100}
    ),
    AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"index": 0, "id": None, "name": None, "args": ' "Python"}', ...}
        ],
        usage_metadata={"input_tokens": 0, "output_tokens": 50, "total_tokens": 50}
    ),
    AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"index": 0, "id": None, "name": None, "args": '}', ...}
        ],
        usage_metadata={"input_tokens": 0, "output_tokens": 50, "total_tokens": 50}
    ),
]

final = sum(chunks, AIMessageChunk(content=""))
# final.tool_call_chunks[0]["args"] == '{"query": "Python"}'
# final.tool_calls == [{"id": "call_1", "name": "search", "args": {"query": "Python"}}]
# final.usage_metadata == {"input_tokens": 100, "output_tokens": 100, "total_tokens": 200}
```

## Utility Functions

Core message utilities (in `langchain_core.messages.utils`):

- **`convert_to_messages(input)`**: Normalizes strings, dicts, or message-like tuples to a list of `BaseMessage` objects.
- **`convert_to_openai_messages(messages)`**: Converts LangChain messages to OpenAI format.
- **`filter_messages(messages, ...)`**: Filters messages by type, id, or name.
- **`trim_messages(messages, ...)`**: Truncates messages to fit token budget (requires `token_counter` callable).
- **`merge_message_runs(messages)`**: Consolidates consecutive messages from the same sender.
- **`get_buffer_string(messages)`**: Formats messages as a human-readable string buffer.
- **`message_chunk_to_message(chunk)`**: Converts a message chunk back to a full message.

## Invariants and Failure Modes

- **Message type immutability**: Once created, a message's `type` field does not change; it is fundamental to serialization.
- **Content coercion**: Non-string content in certain message types (e.g., `ToolMessage`) is coerced to string at creation time; invalid content raises `ValueError`.
- **Tool call ID association**: A `ToolMessage` must have a `tool_call_id` matching an `AIMessage` tool call ID; mismatch is a usage error.
- **Chunk type consistency**: When concatenating chunks, differing types (e.g., `ChatMessageChunk` with different roles) raise `ValueError`.
- **ID handling in streaming**: `None` IDs in early chunks are overridden by non-`None` IDs in later chunks; this enables incremental ID assignment.

## Design and Extension

### BaseMessage as Serializable

Messages inherit from `Serializable`, enabling:
- Round-trip via `dumpd()` / `load()` (LangChain's serialization protocol).
- LangSmith integration for tracing.
- Automatic namespace registration (`["langchain", "schema", "messages"]`).

### Content Blocks as TypedDict

`ContentBlock` is a union of multiple `TypedDict` types, providing:
- Static type checking with full IDE support.
- No runtime cost (TypedDicts are plain dicts at runtime).
- Extensibility: new block types added as new TypedDicts in the union.

### Provider Translator Pattern

Block translators decouple message structure from provider APIs:
- **Responsibility**: Each translator converts an `AIMessage` (or `AIMessageChunk`) to a list of standard `ContentBlock` objects.
- **Registration**: `PROVIDER_TRANSLATORS` dict maps provider names to translator functions.
- **Fallback**: If no translator is registered or translation raises `NotImplementedError`, best-effort parsing applies.

### External Extension

Custom message types can extend `BaseMessage` and override `type` with a custom literal:

```python
class MyCustomMessage(BaseMessage):
    type: Literal["my_custom"] = "my_custom"
    custom_field: str = "..."
```

Custom content blocks are supported via `NonStandardContentBlock` and the `extras` field in all standard blocks.

## Message Type Exports and Public API

The primary entry point for message types and utilities is **`langchain_core.messages`**. The module exports all standard message classes, content block types, serialization functions, utility functions, and the new notification metadata utilities:

**Standard message types**:
- `BaseMessage`, `BaseMessageChunk`
- `HumanMessage`, `HumanMessageChunk`
- `AIMessage`, `AIMessageChunk`
- `SystemMessage`, `SystemMessageChunk`
- `ToolMessage`, `ToolMessageChunk`
- `FunctionMessage`, `FunctionMessageChunk`
- `ChatMessage`, `ChatMessageChunk`
- `RemoveMessage`

**Tool-related types**:
- `ToolCall`, `ToolCallChunk`, `InvalidToolCall`

**Content block types**:
- `ContentBlock` (union type)
- `TextContentBlock`, `ImageContentBlock`, `AudioContentBlock`, `VideoContentBlock`, `FileContentBlock`, `PlainTextContentBlock`, `ReasoningContentBlock`, `DataContentBlock`
- `ServerToolCall`, `ServerToolCallChunk`, `ServerToolResult`
- `NonStandardContentBlock`, `Annotation`, `Citation`, `NonStandardAnnotation`

**Serialization and utilities**:
- `message_to_dict()`, `messages_to_dict()`, `messages_from_dict()`, `_message_from_dict()`
- `merge_content()`, `merge_message_runs()`, `filter_messages()`, `trim_messages()`
- `convert_to_messages()`, `convert_to_openai_messages()`, `get_buffer_string()`, `message_chunk_to_message()`

**Notification metadata** (added in langchain-core 1.7.2):
- `add_notification_metadata()`, `get_notification_metadata()`, `NotificationMetadata`

**Provider translation**:
- `convert_to_openai_data_block()`, `convert_to_openai_image_block()`

**Token and metadata types**:
- `UsageMetadata`, `InputTokenDetails`, `OutputTokenDetails`
- `AnyMessage`, `MessageLikeRepresentation`
