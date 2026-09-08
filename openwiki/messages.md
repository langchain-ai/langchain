---
type: "Architecture"
title: "Message Types and Content Representation"
description: "Document the message abstraction, standardized content blocks for multimodal LLM I/O, message hierarchy, and provider-specific block translators."
tags: [messages, content-blocks, chat-models, streaming, multimodal, provider-adapters]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-77dc1fb726463969f9d53658
    resource: repo://libs/core/langchain_core/messages/ai.py
  - id: openwiki-source-b32b84365d17276620c41ebc
    resource: repo://libs/core/langchain_core/messages/base.py
  - id: openwiki-source-2e77747f30fe980d17d5d1a2
    resource: repo://libs/core/langchain_core/messages/block_translators/__init__.py
  - id: openwiki-source-b0f0ae0889f60e428f2f1b96
    resource: repo://libs/core/langchain_core/messages/block_translators/anthropic.py
  - id: openwiki-source-ad04883edeb0ba80d9ebcb7e
    resource: repo://libs/core/langchain_core/messages/block_translators/langchain_v0.py
  - id: openwiki-source-ac2e0f8b0fb1cb3b223672b7
    resource: repo://libs/core/langchain_core/messages/block_translators/openai.py
  - id: openwiki-source-fc874ddb29b9c5840565397f
    resource: repo://libs/core/langchain_core/messages/content.py
  - id: openwiki-source-8bb392f5dbc1fe7faaf52430
    resource: repo://libs/core/langchain_core/messages/human.py
  - id: openwiki-source-dad8cfeb38a829e03e165986
    resource: repo://libs/core/langchain_core/messages/system.py
  - id: openwiki-source-9861ba5cf0c42c142cf732f9
    resource: repo://libs/core/langchain_core/messages/tool.py
  - id: openwiki-source-498a9586e021b126ab8a8b42
    resource: repo://libs/core/langchain_core/messages/utils.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
---

## Overview

LangChain's message abstraction provides a unified, provider-agnostic interface for representing conversational inputs and outputs to large language models. At its core is **`BaseMessage`**, a serializable container for content that can hold either plain text strings or a structured list of **content blocks**—TypedDict objects representing text, images, audio, video, tool calls, reasoning, and more.

The key innovation is **content blocks**: instead of provider-specific schemas (OpenAI's `image_url` vs. Anthropic's `document` source blocks), LangChain normalizes all content into a unified format. This allows applications to work with multimodal messages portably, with adapters (block translators) converting to provider-specific formats only at invocation time.

## BaseMessage Hierarchy and Core Fields

**Location**: `repo://libs/core/langchain_core/messages/base.py#L93-L180`

`BaseMessage` is the abstract base for all message types. Key fields:

- **`content`**: `str | list[str | dict[Any, Any]]`  
  Holds either plain text or a mixed list of strings (treated as text blocks) and dictionaries (content block dicts).

- **`type`**: `str` (field required by schema)  
  Uniquely identifies the message kind (`"human"`, `"ai"`, `"system"`, `"tool"`, `"chat"`, `"function"`, or chunk variants).

- **`additional_kwargs`**: `dict[Any, Any]`  
  Reserved for provider-specific data not yet mapped to standard fields (e.g., `reasoning_content` from Ollama or DeepSeek).

- **`response_metadata`**: `dict[Any, Any]`  
  Metadata about the response: headers, token counts, model name, provider name, output version.

- **`name`** (optional): Human-readable identifier for the message; unused by most models.

- **`id`** (optional): Unique identifier, typically assigned by the model provider.

### Core Message Types

**`HumanMessage`** (`repo://libs/core/langchain_core/messages/human.py#L9-L61`)  
Represents user input. Used for prompts, questions, and conversation turns from the user. Has chunk variant `HumanMessageChunk` for streaming support.

**`AIMessage`** (`repo://libs/core/langchain_core/messages/ai.py#L160-L305`)  
Represents model output. Contains specialized fields:
- **`tool_calls`**: List of `ToolCall` dicts (structured tool invocation requests).
- **`invalid_tool_calls`**: `ToolCall` dicts that failed parsing (malformed JSON args, etc.).
- **`usage_metadata`**: `UsageMetadata` dict with standardized token counts (`input_tokens`, `output_tokens`, `total_tokens`, plus optional per-category breakdowns).

The `AIMessageChunk` variant is used when streaming and holds `tool_call_chunks` instead of complete `tool_calls`.

**`SystemMessage`** (`repo://libs/core/langchain_core/messages/system.py#L9-L61`)  
Primes model behavior; typically the first message in a conversation. Also supports chunks via `SystemMessageChunk`.

**`ToolMessage`** (`repo://libs/core/langchain_core/messages/tool.py#L26-L164`)  
Represents the result of a tool invocation. Required fields:
- **`tool_call_id`**: Links this result to the `AIMessage.tool_calls[].id` that requested it.
- **`content`**: The tool's output (string or list of content blocks).
- **`status`**: `"success"` or `"error"`.
- **`artifact`** (optional): Full tool output not sent to model (e.g., raw data when only a summary is in `content`).

**`ChatMessage`** and **`FunctionMessage`**  
Legacy/specialized message types. `ChatMessage` is generic with a `role` field; `FunctionMessage` represents deprecated function-calling format.

### Message Chunks and Streaming

**Location**: `repo://libs/core/langchain_core/messages/base.py#L450-L500` (BaseMessageChunk)

During streaming, models emit `AIMessageChunk` objects incrementally. These chunks are designed to be **mergeable**: when combined with `+`, they accumulate content, merge tool call chunks by index, and aggregate token usage.

**`AIMessageChunk`** fields:
- **`tool_call_chunks`**: Partial tool call objects with nullable `name` and `args` (JSON string fragments).
- **`chunk_position`**: `"last"` on the final chunk, signaling aggregation triggers (e.g., parsing completed tool call chunks into full `tool_calls`).

Merging chunks with `+` invokes `add_ai_message_chunks()`, which:
- Concatenates string content and merges list content blocks.
- Combines `tool_call_chunks`, respecting their `index` field.
- Aggregates token usage across chunks.
- On the final chunk (`chunk_position="last"`), parses accumulated tool call chunks into complete `ToolCall` objects.

## Content Blocks: Unified Multimodal Representation

**Location**: `repo://libs/core/langchain_core/messages/content.py#L1-L878`

Content blocks are **TypedDict objects** representing distinct types of message content. They provide a provider-agnostic abstraction that block translators convert to provider formats.

### Standard Block Types

**TextContentBlock**  
```python
{
    "type": "text",
    "text": str,
    "id": str (optional, auto-generated),
    "annotations": list[Annotation] (optional, citations/metadata),
    "index": int | str (optional, for streaming),
    "extras": dict (optional, provider-specific fields),
}
```
Plain text output from a model. Annotations enable citations pointing to source documents.

**ReasoningContentBlock**  
```python
{
    "type": "reasoning",
    "reasoning": str (optional),
    "id": str (optional),
    "index": int | str (optional),
    "extras": dict (optional),
}
```
Chain-of-thought or intermediate reasoning from models like o1, o3, etc. Often extracted from `<think>` tags or provider-specific fields in `additional_kwargs`.

**ToolCall**  
```python
{
    "type": "tool_call",
    "id": str | None,
    "name": str,
    "args": dict,
    "index": int | str (optional),
    "extras": dict (optional),
}
```
A request from the model to invoke a tool. ID must be unique per message to match with `ToolMessage` responses.

**ToolCallChunk** (streaming variant)  
```python
{
    "type": "tool_call_chunk",
    "id": str | None,
    "name": str | None,
    "args": str | None,
    "index": int | str (optional),
    "extras": dict (optional),
}
```
Partial tool call (emitted when streaming). String `args` accumulates JSON. Chunks with the same `index` are merged on arrival.

**InvalidToolCall**  
```python
{
    "type": "invalid_tool_call",
    "id": str | None,
    "name": str | None,
    "args": str | None,
    "error": str | None,
    "index": int | str (optional),
    "extras": dict (optional),
}
```
Tool call that failed parsing. Error field captures the exception message.

### Multimodal Data Blocks

**ImageContentBlock**  
```python
{
    "type": "image",
    "url": str (optional),
    "base64": str (optional),
    "file_id": str (optional),
    "mime_type": str (optional, required for base64),
    "id": str (optional),
    "index": int | str (optional),
    "extras": dict (optional),
}
```
Image data via URL, base64 encoding, or cloud file reference (e.g., OpenAI Files API).

**AudioContentBlock**, **VideoContentBlock**  
Similar structure to `ImageContentBlock` with `type` fields `"audio"` and `"video"`.

**FileContentBlock**  
```python
{
    "type": "file",
    "url": str (optional),
    "base64": str (optional),
    "file_id": str (optional),
    "mime_type": str (optional),
    "id": str (optional),
    "index": int | str (optional),
    "extras": dict (optional),
}
```
Generic file data (PDFs, Word docs, etc.) not covered by image/audio/plaintext types.

**PlainTextContentBlock**  
```python
{
    "type": "text-plain",
    "text": str (optional),
    "base64": str (optional),
    "url": str (optional),
    "file_id": str (optional),
    "mime_type": Literal["text/plain"],
    "title": str (optional),
    "context": str (optional),
    "id": str (optional),
    "index": int | str (optional),
    "extras": dict (optional),
}
```
Plain text documents with optional title and context for model interpretation.

### Server-Side Tool Calls

**ServerToolCall**, **ServerToolCallChunk**, **ServerToolResult**  
Support tool execution that happens server-side (e.g., code execution, web search). Models emit these to request execution without local handler code.

### NonStandardContentBlock

```python
{
    "type": "non_standard",
    "value": dict,
    "id": str (optional),
    "index": int | str (optional),
}
```
Holds provider-specific content that doesn't map to standard block types. Block translators attempt to parse non-standard blocks during the `content_blocks` property evaluation.

### Accessing Content Blocks

**Location**: `repo://libs/core/langchain_core/messages/base.py#L199-L260`

The `content_blocks` property normalizes message content to a list of typed content block dicts:

```python
@property
def content_blocks(self) -> list[types.ContentBlock]:
```

**Behavior**:
1. If `content` is a string, wrap it as `{"type": "text", "text": content}`.
2. Parse list items: strings become text blocks, dicts with known `type` values are kept as-is, others become `{"type": "non_standard", "value": ...}`.
3. For `AIMessage`, check `response_metadata["model_provider"]` and use the provider's translator if registered (e.g., OpenAI, Anthropic).
4. Fall back to best-effort parsing if no translator exists.
5. For `AIMessage`, append `tool_calls` not already in content as tool call blocks.
6. Extract reasoning from `additional_kwargs["reasoning_content"]` if present.

## Block Translators: Adapting to Provider Formats

**Location**: `repo://libs/core/langchain_core/messages/block_translators/__init__.py` and provider modules

Block translators convert between LangChain's standard blocks and provider-specific formats. Each provider module registers translator functions that are invoked when accessing `AIMessage.content_blocks` if `response_metadata["model_provider"]` matches.

### Registration System

**`register_translator`** and **`get_translator`**:
```python
def register_translator(
    provider: str,
    translate_content: Callable[[AIMessage], list[ContentBlock]],
    translate_content_chunk: Callable[[AIMessageChunk], list[ContentBlock]],
) -> None
```

Translators are stored in `PROVIDER_TRANSLATORS` and auto-initialized on module load via `_register_translators()`.

### Key Translators

**OpenAI** (`repo://libs/core/langchain_core/messages/block_translators/openai.py`)  
Handles Chat Completions format:
- Converts OpenAI's `image_url` blocks to standard `ImageContentBlock`.
- Parses `tool_calls` (from function calling) into `ToolCall` blocks.
- Supports Responses API with `input_audio`, `input_file`, and `input_image` types.
- `convert_to_openai_image_block()` and `convert_to_openai_data_block()` are public utilities used by models and integrations.

**Anthropic** (`repo://libs/core/langchain_core/messages/block_translators/anthropic.py`)  
Handles Anthropic's format:
- Converts `document` blocks (with `source` field specifying type: `base64`, `url`, `file`, or `text`) to standard file/plaintext blocks.
- Converts `image` blocks with various source types to `ImageContentBlock`.
- Populates `extras` with provider-specific fields like `cache_control`.

**Google GenAI** and **Bedrock Converse**  
Similar conversion logic for Google and AWS formats.

**LangChain v0 (Backward Compatibility)** (`repo://libs/core/langchain_core/messages/block_translators/langchain_v0.py`)  
Parses legacy `source_type`-based blocks (e.g., `{"type": "image", "source_type": "url", "url": "..."}`) into v1 blocks.

### Translation Flow in `content_blocks`

When `AIMessage.content_blocks` is accessed:
1. Check `response_metadata["output_version"]` for `"v1"` (already normalized, short-circuit).
2. Attempt provider-specific translation if `response_metadata["model_provider"]` is set.
3. Fall back to `BaseMessage.content_blocks` best-effort parsing.
4. For `AIMessage` specifically, append any tool calls not already in content and extract reasoning from kwargs.

## Message Manipulation Utilities

**Location**: `repo://libs/core/langchain_core/messages/utils.py#L1-L150` and beyond

The utils module provides helpers for working with messages:

**`get_buffer_string`**  
Converts a sequence of messages to a single string for logging/debugging. Supports `format="prefix"` (role-prefixed) or `format="xml"` (XML tags with proper escaping). Multimodal content blocks are rendered with safe truncation and base64-encoded data omitted.

**`convert_to_messages` and `convert_to_openai_messages`**  
Coerce various input formats (dicts, strings, `MessageLikeRepresentation` union) into typed message objects.

**`filter_messages`, `trim_messages`, `merge_message_runs`**  
Filter, truncate, and deduplicate consecutive messages of the same type.

**`AnyMessage` Union Type**  
```python
AnyMessage = Annotated[
    Annotated[AIMessage, Tag(tag="ai")]
    | Annotated[HumanMessage, Tag(tag="human")]
    | ... (all message and chunk types)
    Field(discriminator=Discriminator(_get_type)),
]
```
A tagged union for Pydantic deserialization. The `type` field discriminates the correct message class during deserialization.

## Content Representation: String vs. Block List

Messages accept content in two forms:

**String content**:
```python
AIMessage(content="Hello, world!")
```
Simple, backward compatible. Treated internally as a single text block.

**Block list content**:
```python
AIMessage(
    content=[
        {"type": "text", "text": "What is this?"},
        {"type": "image", "url": "https://example.com/img.png"},
    ]
)
```
or using the typed `content_blocks` kwarg:
```python
AIMessage(
    content_blocks=[
        create_text_block("What is this?"),
        create_image_block(url="https://example.com/img.png"),
    ]
)
```

The `text` property extracts all text blocks:
```python
msg = AIMessage(content=[
    {"type": "text", "text": "Hello"},
    {"type": "image", "url": "..."},
])
print(msg.text)  # "Hello"
```

## Integration with Chat Models

Chat models normalize message input and output using the message abstraction:

- **Input**: Users provide messages (strings, dicts, or `MessageLikeRepresentation`). Models invoke `_normalize_messages()` to convert to `BaseMessage` objects and optionally expand multimodal content for the target provider.

- **Output**: Models return `AIMessage` with:
  - `content`: Model's text response (or list of blocks if multimodal).
  - `response_metadata`: Populated with `model_provider`, `output_version`, token counts, etc.
  - `tool_calls`: Parsed from provider format into structured `ToolCall` dicts.
  - `usage_metadata`: Standardized token counts.

See `/openwiki/chat-models.md` for details on model invocation and streaming lifecycle.

## Provider-Specific Extensions

The `extras` field in content blocks allows provider metadata without breaking standard structure:

```python
{
    "type": "text",
    "text": "Response text",
    "extras": {
        "thought_signature": "EpoWCpc...",  # Google
        "cache_control": {"type": "ephemeral"},  # Anthropic
    },
}
```

This approach maintains type safety while supporting emerging provider capabilities.

## Backward Compatibility and Versioning

LangChain v1.0 introduced the v1 content block format, superseding the v0 `source_type` style. The block translators handle both:

- **v0 blocks** (e.g., `{"type": "image", "source_type": "url", "url": "..."}`) are recognized and wrapped as non-standard blocks, then parsed by `_convert_v0_multimodal_input_to_v1()`.
- **Provider-specific blocks** (e.g., OpenAI's `image_url` from raw API responses) are unpacked by provider translators.
- **Output version tracking**: `response_metadata["output_version"] = "v1"` signals that content is already normalized, allowing short-circuit optimization.

## Example Workflows

### Sending a Multimodal Message

```python
from langchain_core.messages import HumanMessage, create_text_block, create_image_block

message = HumanMessage(
    content_blocks=[
        create_text_block("Describe this chart."),
        create_image_block(url="https://example.com/chart.png", mime_type="image/png"),
    ]
)

# Access text
print(message.text)  # "Describe this chart."

# Get normalized blocks
for block in message.content_blocks:
    print(block["type"])  # "text", "image"
```

### Handling Tool Calls from a Model

```python
ai_msg = model.invoke([...])
# ai_msg.tool_calls = [
#     {"type": "tool_call", "id": "call_1", "name": "search", "args": {"query": "..."}},
# ]

for tool_call in ai_msg.tool_calls:
    result = invoke_tool(tool_call["name"], tool_call["args"])
    tool_response = ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"],
    )
```

### Streaming and Chunk Aggregation

```python
chunks = []
for chunk in model.stream(input_msg):
    chunks.append(chunk)
    print(f"Received: {chunk.content}")

# Aggregate all chunks
final = chunks[0]
for chunk in chunks[1:]:
    final = final + chunk

# final.tool_calls are now complete (parsed from tool_call_chunks)
```

### Using Block Translators

Block translators are invoked transparently when a model sets `response_metadata["model_provider"]`:

```python
# OpenAI model
ai_msg = openai_model.invoke(msg)
# response_metadata contains model_provider="openai"

blocks = ai_msg.content_blocks
# If content is from OpenAI's API, translator converts image_url → ImageContentBlock
```

Custom provider integrations can register their own translator:

```python
from langchain_core.messages.block_translators import register_translator

def my_translate_content(msg: AIMessage) -> list[ContentBlock]:
    # Custom logic
    pass

def my_translate_content_chunk(chunk: AIMessageChunk) -> list[ContentBlock]:
    # Custom logic
    pass

register_translator("my_provider", my_translate_content, my_translate_content_chunk)
```
