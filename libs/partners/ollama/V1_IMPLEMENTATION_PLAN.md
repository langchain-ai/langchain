# Ollama Provider V1 Content Support Implementation Plan

This document outlines the detailed implementation plan for adding v1 content (standard block types) support to Ollama, following the patterns established in the OpenAI provider implementation.

## Current State Analysis

Based on detailed analysis of the ChatOllama implementation at `libs/partners/ollama/langchain_ollama/chat_models.py`, focusing on the ChatOllama class.

### ChatOllama Architecture

**Core Structure:**

- Inherits from `BaseChatModel`
- Located at `langchain_ollama/chat_models.py:215`
- Main message conversion: `_convert_messages_to_ollama_messages()`

### Current Content Support (v0 format)

**Input Message Processing (_convert_messages_to_ollama_messages):**

- **Text content:** Direct string content or `{"type": "text", "text": "..."}` blocks
- **Images:** `{"type": "image_url"}` blocks and `is_data_content_block()` (v1 format already supported via line 691)
- **Tool calls:** Via `message.tool_calls` property, converted to OpenAI format
- **Reasoning:** Currently handled in `additional_kwargs`, not in content blocks

**Output Message Generation:**

- **Text content:** String format in `AIMessage.content`
- **Tool calls:** In `message.tool_calls` property
- **Reasoning:** Stored in `additional_kwargs["reasoning_content"]` when `reasoning=True`
- **Stream processing:** `_iterate_over_stream()` and `_aiterate_over_stream()`
- **Non-streaming:** `_generate()` and `_agenerate()`

### Key Implementation Requirements for V1 Conversion

**Output Conversion (Ollama → v1):**

```python
# Current v0 format
AIMessage(
    content="Hello world",
    tool_calls=[{"name": "search", "args": {...}, "id": "123"}],
    additional_kwargs={"reasoning_content": "I need to search..."}
)

# Target v1 format (typed Dicts)
AIMessage(
    content=[
        ReasoningContentBlock(type="reasoning", reasoning="I need to search..."),
        TextContentBlock(type="text", text="Hello world"),
        ToolCallContentBlock(type="tool_call", id="123")
    ]
)
```

**Input Processing (v1 → Ollama):**

- Extract text from `{"type": "text"}` blocks for content string
- Extract reasoning from `{"type": "reasoning"}` blocks for API `reasoning` parameter
  <!-- When processing v1 input, reasoning content blocks need to be converted to the `reasoning=True` parameter that Ollama's API expects, rather than being passed as content blocks -->
- Skip `{"type": "tool_call"}` blocks (handled by `tool_calls` property)
  <!-- Tool call blocks in v1 content are metadata - the actual tool calls are handled via the separate `tool_calls` property on the message, so we don't include these blocks in the content string sent to Ollama -->
- Handle existing `{"type": "image"}` data blocks (already supported via `_get_image_from_data_content_block:199`)
  <!-- Image blocks are already processed correctly by the existing `is_data_content_block()` check and `_get_image_from_data_content_block()` function, so v1 image blocks will work without changes -->

**Critical Integration Points:**

1. `_iterate_over_stream()` and `_aiterate_over_stream()`: Add conditional v1 conversion
2. `_generate()` and `_agenerate()`: Add conditional v1 conversion for final message  
3. `_convert_messages_to_ollama_messages()`: Handle v1 input format unconditionally

**Patterns Following OpenAI Implementation:**

- **Conditional Output Conversion:** Only convert to v1 when `output_version="v1"`
- **Unconditional Input Processing:** Handle v1 input regardless of output setting
- **Separate Compatibility Module:** Create `_compat.py` with conversion functions

## Implementation Tasks

### Phase 1: Core Infrastructure

#### 1.1 Add `output_version` Field to ChatOllama

**File:** `langchain_ollama/chat_models.py`

Add field to ChatOllama class:

```python
output_version: str = "v0"
"""Version of AIMessage output format to use.

This field is used to roll-out new output formats for chat model AIMessages
in a backwards-compatible way.

Supported values:

- ``"v0"``: AIMessage format as of langchain-ollama 0.x.x.
- ``"v1"``: v1 of LangChain cross-provider standard.

``output_version="v1"`` is recommended.

.. versionadded:: 0.4.0
"""
```

#### 1.2 Create Compatibility Module

**File:** `langchain_ollama/chat_models/_compat.py`

Create conversion functions following the patterns introduced in `langchain-openai`, including LangChain v0.3 compatibility for users migrating stored/cached AIMessage objects.

**Core conversion functions:**

- `_convert_to_v1_from_ollama_format(message: AIMessage) -> AIMessageV1` - Ollama native → v1
- `_convert_to_v1_from_ollama_chunk(chunk: AIMessageChunk) -> AIMessageChunkV1` - Ollama native → v1 (chunk conversion)
- `_convert_from_v1_to_ollama_format(message: AIMessageV1) -> AIMessage` - v1 → Ollama native (for requests)

**LangChain v0.3 compatibility functions:**

- `_convert_from_v03_ai_message(message: AIMessage) -> AIMessageV1` - LangChain v0.3 → v1

### Phase 2: Message Conversion Functions

#### 2.1 Encoding Functions (v1 → Ollama)

```python
def _convert_from_v1_to_ollama_format(message: AIMessageV1) -> AIMessage:
    """Convert a v1 message to Ollama-compatible request format.
    
    Always called during input processing, regardless of ``output_version`` setting.
    Returns AIMessage with v0-style string content and reasoning in ``additional_kwargs``.
    If input is already v0 format, returns unchanged.
    """
    if isinstance(message.content, list):
        # Extract text content for Ollama API
        text_content = ""
        reasoning_content = None
        
        for block in message.content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    text_content += block.get("text", "")
                elif block_type == "reasoning":
                    # Extract reasoning for API parameter
                    reasoning_content = block.get("reasoning", "")
                elif block_type == "tool_call":
                    # Skip - handled via tool_calls property
                    continue
                elif is_data_content_block(block):
                    # Keep data blocks as-is (images already supported)
                    continue
        
        # Update message with extracted content
        updates = {"content": text_content if text_content else ""}
        if reasoning_content:
            additional_kwargs = dict(message.additional_kwargs)
            additional_kwargs["reasoning_content"] = reasoning_content
            updates["additional_kwargs"] = additional_kwargs
            
        return message.model_copy(update=updates)
    return message # Not a v1 message, return as is
```

#### 2.2 Decoding Functions (Ollama → v1)

```python
def _convert_to_v1_from_ollama_format(message: AIMessage) -> AIMessage:
    """Convert an Ollama message to v1 format."""
    new_content: list = []
    
    # Handle reasoning content first (from additional_kwargs)
    additional_kwargs = dict(message.additional_kwargs)
    if reasoning_content := additional_kwargs.pop("reasoning_content", None):
        new_content.append({
            ReasoningContentBlock(type="reasoning", reasoning=reasoning_content)
        })
    
    # Convert text content to content blocks
    if isinstance(message.content, str) and message.content:
        new_content.append({
            TextContentBlock(type="text", text=message.content)
        })
    
    # Convert tool calls to content blocks
    for tool_call in message.tool_calls:
        if id_ := tool_call.get("id"):
            new_content.append(
                ToolCallContentBlock(type="tool_call", id=id_)
            )
    
    return message.model_copy(update={
        "content": new_content,
        "additional_kwargs": additional_kwargs
    })
```

#### 2.3 Streaming Chunk Conversion

```python
def _convert_to_v1_from_ollama_chunk(chunk: AIMessageChunk) -> AIMessageChunk:
    """Convert an Ollama chunk to v1 format."""
    result = _convert_to_v1_from_ollama_format(cast(AIMessage, chunk))
    return cast(AIMessageChunk, result)
```

### Phase 3: Integration Points

#### 3.1 Update Message Generation

**Location:** `ChatOllama._iterate_over_stream()` and `ChatOllama._aiterate_over_stream()`

Add v1 conversion logic following OpenAI's pattern of **conditional conversion**:

```python
# In _iterate_over_stream method, after creating chunk:
chunk = ChatGenerationChunk(
    message=AIMessageChunk(
        content=content,
        additional_kwargs=additional_kwargs,
        usage_metadata=_get_usage_metadata_from_generation_info(stream_resp),
        tool_calls=_get_tool_calls_from_response(stream_resp),
    ),
    generation_info=generation_info,
)

# Add v1 conversion - ONLY when output_version == "v1"
# Following OpenAI pattern: conversion is conditional, not always applied
if self.output_version == "v1":
    chunk.message = _convert_to_v1_from_ollama_chunk(
        cast(AIMessageChunk, chunk.message)
    )

yield chunk
```

#### 3.2 Update Non-Streaming Generation

**Location:** `ChatOllama._generate()` and `ChatOllama._agenerate()`

Add v1 conversion following the same conditional pattern:

```python
# After creating AIMessage:
ai_message = AIMessage(
    content=final_chunk.text,
    usage_metadata=cast(AIMessageChunk, final_chunk.message).usage_metadata,
    tool_calls=cast(AIMessageChunk, final_chunk.message).tool_calls,
    additional_kwargs=final_chunk.message.additional_kwargs,
)

# Add v1 conversion - ONLY when output_version == "v1" 
# v0 format (native Ollama) requires no conversion
if self.output_version == "v1":
    ai_message = _convert_to_v1_from_ollama_format(ai_message)

chat_generation = ChatGeneration(
    message=ai_message,
    generation_info=generation_info,
)
```

### Phase 4: Enhanced Content Block Support

#### 4.1 Data Content Block Handling

Update `_get_image_from_data_content_block()` to handle v1 format:

```python
def _get_image_from_data_content_block(block: dict) -> str:
    """Format v1 data content block to Ollama format."""
    if block["type"] == "image":
        if "base64" in block:
            return block["base64"]
        elif block.get("source_type") == "base64" and "data" in block:
            return block["data"]
        else:
            raise ValueError("Image data only supported through base64 format.")
    
    raise ValueError(f"Blocks of type {block['type']} not supported.")
```

#### 4.2 Update Message Conversion

Enhance `_convert_messages_to_ollama_messages()` to handle v1 content:

```python
# In the content processing loop:
for content_part in cast(list[dict], message.content):
    if content_part.get("type") == "text":
        content += f"\n{content_part['text']}"
    elif content_part.get("type") == "tool_call":
        # Skip - handled by tool_calls property
        continue
    elif content_part.get("type") == "reasoning":
        # Skip - handled by reasoning parameter
        continue
    elif is_data_content_block(content_part):
        # Handle v1 data blocks
        if content_part["type"] == "image":
            image = _get_image_from_data_content_block(content_part)
            images.append(image)
    # ... existing logic
```

### Phase 5: Reasoning Integration

#### 5.1 V1 Reasoning Block Support

Handle reasoning content blocks in v1 format:

```python
def _extract_reasoning_from_v1_content(message: AIMessage) -> Optional[str]:
    """Extract reasoning content from v1 message content blocks."""
    if not isinstance(message.content, list):
        return None
    
    for block in message.content:
        if isinstance(block, dict) and block.get("type") == "reasoning":
            return block.get("reasoning", "")
    return None
```

#### 5.2 Update Message Processing for Input

**Key Insight**: Unlike output conversion which is conditional on `output_version`, **input message processing must handle v1 messages regardless of the output_version setting**. Users might pass v1 format messages even when expecting v0 output.

Modify `_convert_messages_to_ollama_messages()` to handle v1 input:

```python
def _convert_messages_to_ollama_messages(self, messages: list[BaseMessage]) -> list[dict]:
    """Convert messages to Ollama format, handling both v0 and v1 input formats."""
    ollama_messages = []
    for message in messages:
        # Handle v1 format messages in input (regardless of output_version)
        if isinstance(message, AIMessage) and isinstance(message.content, list):
            # This is likely a v1 format message - convert for Ollama API
            converted_message = _convert_from_v1_to_ollama_format(message)
            ollama_message = self._convert_single_message_to_ollama(converted_message)
        else:
            # v0 format or other message types - process normally
            ollama_message = self._convert_single_message_to_ollama(message)
        ollama_messages.append(ollama_message)
    
    return ollama_messages
```

**Pattern**: Input processing handles both formats, output processing is conditional on `output_version`.

### Phase 6: Testing and Validation

#### 6.1 Unit Tests

**File:** `tests/unit_tests/test_chat_models_v1.py`

Create comprehensive tests:

- Test v1 format conversion (text, tool calls, reasoning)
- Test backwards compatibility with v0
- Test streaming vs non-streaming consistency
- Test data content block handling

#### 6.2 Integration Tests

**File:** `tests/integration_tests/test_chat_models_v1.py`

Test with real Ollama instances:

- Standard chat functionality with v1 format
- Tool calling with v1 format
- Reasoning models with v1 format
- Mixed content (text + images) with v1 format

### Phase 7: Documentation and Migration

#### 7.1 Update Docstrings

Add v1 format examples to ChatOllama class docstring:

```python
"""
V1 Output Format:
    .. code-block:: python

        llm = ChatOllama(model="llama3.1", output_version="v1")
        response = llm.invoke("Hello")
        
        # Response content is now a list of content blocks:
        response.content
        # [{"type": "text", "text": "Hello! How can I help you?"}]
"""
```

#### 7.2 Migration Guide

Create documentation for migrating from v0 to v1:

- Breaking changes (content format)
- Benefits of v1 format
- Migration timeline and recommendations

## Implementation Checklist

### Core Implementation

- [ ] Add `output_version` field to `ChatOllama`
- [ ] Create `_compat.py` module with conversion functions
- [ ] Implement `_convert_to_v1_from_ollama_format()` - for output conversion
- [ ] Implement `_convert_from_v1_to_ollama_format()` - for input processing
- [ ] Implement chunk conversion functions
- [ ] Implement `_convert_to_v03_ai_message()` - for LangChain v0.3 compatibility
- [ ] Implement `_convert_from_v03_ai_message()` - for LangChain v0.3 migration

### Integration

- [ ] Update `_iterate_over_stream()` for conditional v1 output conversion
- [ ] Update `_aiterate_over_stream()` for conditional v1 output conversion
- [ ] Update `_generate()` for conditional v1 output conversion
- [ ] Update `_agenerate()` for conditional v1 output conversion
- [ ] Update `_convert_messages_to_ollama_messages()` for v1 input handling (unconditional)

### Content Blocks

- [ ] Enhance data content block handling
- [ ] Add reasoning block support
- [ ] Update tool call handling for v1

### Testing

- [ ] Create unit tests for all conversion functions
- [ ] Create integration tests for v1 functionality
- [ ] Test backwards compatibility

### Documentation

- [ ] Update class docstrings with v1 examples (reStructuredText)
- [ ] Create migration guide

## Dependencies

### Required Imports

Add to `langchain_ollama/chat_models.py`:

```python
from langchain_core.messages import is_data_content_block
```

Add to `langchain_ollama/chat_models/_compat.py`:

```python
from typing import Any, cast
from langchain_core.messages import AIMessage, AIMessageChunk, is_data_content_block
```

## Migration Strategy

### Phase 1

- Core infrastructure and conversion functions
- Basic v1 support without breaking changes

### Phase 2

- Comprehensive test suite
- Edge case handling

### Phase 3

- Complete documentation
- Migration guides

## Risk Mitigation

### Backwards Compatibility

- Default `output_version="v0"` maintains existing behavior
- All existing tests should pass without modification

## Success Criteria

1. **Functional**: All v1 message formats work correctly with Ollama
2. **Compatible**: No breaking changes to existing v0 usage
3. **Documented**: Complete documentation and migration guides
