# Provider V1 Content Support Implementation Plan

Implementation plan for adding messages with standard content block types (v1) to any chat model provider.

## Architecture Overview

### V1 Chat Model Foundation

The v1 architecture uses:

- **Separate File Structure**: Create `base_v1.py` alongside existing `base.py`
- **BaseChatModelV1 Inheritance**: Inherit from `langchain_core.language_models.v1.chat_models.BaseChatModelV1`
- **Native V1 Messages**: Use `AIMessageV1`, `HumanMessageV1`, etc. throughout
- **No `output_version` Field**: Always return v1 format (no conditional logic) - this is a deviation from our prior plan
- **Standardized Content Blocks**: Native support for `TextContentBlock`, `ImageContentBlock`, `ToolCallContentBlock`, etc. (defined in `libs/core/langchain_core/messages/content_blocks.py`)

### Implementation Pattern

Based on successful implementations, we have both:

```python
# Existing v0 implementation (unchanged)
libs/partners/{provider}/langchain_{provider}/chat_models/base.py
class ChatProvider(BaseChatModel):  # v0 base class
    output_version: str = "v0"  # conditional v1 support - REMOVE????

# NEW FILE: v1 implementation
libs/partners/{provider}/langchain_{provider}/chat_models/base_v1.py
class ChatProviderV1(BaseChatModelV1):  # v1 base class
    # Inherits from BaseChatModelV1
```

### V1 Message Flow Pattern

```python
from langchain_core.messages.v1 import HumanMessage as HumanMessageV1
from langchain_core.messages.content_blocks import TextContentBlock, ImageContentBlock

user_message = HumanMessageV1(content=[
    TextContentBlock(type="text", text="Hello"),
    ImageContentBlock(type="image", mime_type="image/jpeg", base64="...")
])

# Internal: Convert v1 to provider's native format
# (For API calls, either direct or via SDK)
provider_request = convert_from_v1_to_provider_api(user_message)
provider_response = provider_api.chat(provider_request)

# Output: Convert provider response to v1 format
ai_message_v1 = convert_to_v1_from_provider_response(provider_response)
```

## Implementation Framework

### Phase 1: Core Infrastructure

#### 1.1 Create Separate V1 File Structure

**New File:** `langchain_{provider}/chat_models/base_v1.py`

```python
"""Provider chat model v1 implementation.

As much as possible, this should be a direct copy of the v0
implementation, but with the following changes:

- Inherit from BaseChatModelV1
- Use native v1 messages (AIMessageV1, HumanMessageV1, etc.)
- `content` is a list of ContentBlock objects (TextContentBlock, ImageContentBlock, etc.)
"""

from langchain_core.language_models.v1.chat_models import (
    BaseChatModelV1,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages.v1 import (
    AIMessage as AIMessageV1,
    AIMessageChunk as AIMessageChunkV1,
    HumanMessage as HumanMessageV1,
    MessageV1,
    SystemMessage as SystemMessageV1,
    ToolMessage as ToolMessageV1,
    ResponseMetadata,
)
from langchain_core.messages import content_blocks as types
from pydantic import Field

class BaseChatProviderV1(BaseChatModelV1):
    """Base class for provider v1 chat models."""

    model_name: str = Field(default="default-model", alias="model")
    """Model name to use."""

    # Provider-specific configuration fields
    # ... (copy from existing base.py but adapt for v1 where applicable)

class ChatProviderV1(BaseChatProviderV1):
    """Provider chat model with v1 messages."""

    def _generate_stream(
        self,
        messages: list[MessageV1],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunkV1]:
        """Generate streaming response with v1 messages."""

    def _agenerate_stream(
        self,
        messages: list[MessageV1],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunkV1]:
        """Generate async streaming response with v1 messages."""
```

#### 1.2 Update Package Exports

Include the new v1 chat model in the package exports:

**File:** `langchain_{provider}/chat_models/__init__.py`

```python
from langchain_{provider}.chat_models.base import ChatProvider
from langchain_{provider}.chat_models.base_v1 import ChatProvider as ChatProviderV1

__all__ = ["ChatProvider", "ChatProviderV1"]
```

#### 1.3 Create Conversion Utilities

**File:** `langchain_{provider}/chat_models/_compat.py`

```python
"""V1 message conversion utilities."""

from typing import Any, Union, cast
from langchain_core.messages.v1 import MessageV1, AIMessageV1
from langchain_core.messages import content_blocks as types

def _convert_from_v1_to_provider_format(message: MessageV1) -> dict[str, Any]:
    """Convert v1 message to provider API format."""

def _convert_to_v1_from_provider_format(response: dict[str, Any]) -> AIMessageV1:
    """Convert provider API response to v1 message(s)."""

def _format_v1_message_content(content: list[ContentBlock]) -> Any:
    """Format v1 content blocks for provider API."""
```

### Phase 2: Message Processing

#### 2.1 Input Message Handling

```python
def _convert_from_v1_to_provider_format(message: MessageV1) -> dict[str, Any]:
    """Convert v1 message to provider API format."""
    if isinstance(message, HumanMessageV1):
        return _convert_human_message_v1(message)
    elif isinstance(message, AIMessageV1):
        return _convert_ai_message_v1(message)
    elif isinstance(message, SystemMessageV1):
        return _convert_system_message_v1(message)
    elif isinstance(message, ToolMessageV1):
        return _convert_tool_message_v1(message)
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")

def _convert_content_blocks_to_provider_format(content: list[ContentBlock]) -> list[dict]:
    """Convert v1 content blocks to provider API format.

    Shared across all message types since they all support the same content blocks.
    """
    content_parts = []

    for block in content:
        block_type = block.get("type")
        if block_type == "text":
            # The format here will vary depending on the provider's API
            content_parts.append({
                "type": "text",
                "text": block.get("text", "")
            })
        elif block_type == "image":
            content_parts.append(_convert_image_block_to_provider(block))
        elif block_type == "audio":
            content_parts.append(_convert_audio_block_to_provider(block))
        elif block_type == "tool_call":
            # Skip tool calls - handled separately via tool_calls property
            continue
        # Add other content block types...

    return content_parts

def _convert_human_message_v1(message: HumanMessageV1) -> dict[str, Any]:
    """Convert HumanMessageV1 to provider format."""
    # The format here will vary depending on the provider's API
    return {
        "role": "user",
        "content": _convert_content_blocks_to_provider_format(message.content),
        "name": message.name,
    }

def _convert_ai_message_v1(message: AIMessageV1) -> dict[str, Any]:
    """Convert AIMessageV1 to provider format."""
    # Extract text content for main content field
    # The format here will vary depending on the provider's API
    text_content = ""
    for block in message.content:
        if block.get("type") == "text":
            text_content += block.get("text", "")

    return {
        "role": "assistant",
        "content": text_content,
        "tool_calls": [_convert_tool_call_to_provider(tc) for tc in message.tool_calls],
        "name": message.name,
    }
```

#### 2.2 Output Message Generation

Convert provider responses directly to v1 format:

```python
from langchain_core.messages.content_blocks import (
    TextContentBlock,
    ToolCall,
    ReasoningContentBlock,
    AudioContentBlock,
    ...
)

def _convert_to_v1_from_provider_format(response: dict[str, Any]) -> AIMessageV1:
    """Convert provider response to AIMessageV1."""
    # The format here will vary depending on the provider's API
    # (This is a dummy implementation)
    content: list[ContentBlock] = []

    if text_content := response.get("content"):
        #  For example, if the text response from the provider comes in as `content`:
        if isinstance(text_content, str) and text_content:
            content.append(TextContentBlock(type="text", text=text_content))
        elif isinstance(text_content, list):
            # If the content is a list of text items
            for item in text_content:
                if item.get("type") == "text":
                    content.append(TextContentBlock(
                        type="text",
                        text=item.get("text", "")
                    ))

    if tool_calls := response.get("tool_calls"):
        # Similarly, if the provider returns tool calls under `tool_calls`:
        for tool_call in tool_calls:
            content.append(ToolCall(
                type="tool_call",
                id=tool_call.get("id", ""),
                name=tool_call.get("function", {}).get("name", ""),
                args=tool_call.get("function", {}).get("arguments", {}),
            ))

    # Some providers call this `reasoning`, `thoughts`, `thinking`
    if reasoning := response.get("reasoning"):
        # May opt to insert reasoning in specific order depending on API design
        content.insert(0, ReasoningContentBlock(
            type="reasoning",
            reasoning=reasoning
        ))

    if audio := response.get("audio"):
        content.append(AudioContentBlock(
            type="audio",
            # Provider-specific fields via PEP 728 TypedDict extra items (common for multimodal)
        ))

    return AIMessageV1(
        content=content,
        response_metadata=ResponseMetadata(
            model_name=response.get("model"),  # or whatever key the provider uses for model name
            usage=response.get("usage", {}),
            # Other provider-specific metadata
        ),
    )
```

### Phase 3: Streaming Implementation

Implement streaming that yields `AIMessageChunkV1` directly:

```python
def _generate_stream(
    self,
    messages: list[MessageV1],
    stop: Optional[list[str]] = None,
    **kwargs: Any,
) -> Iterator[AIMessageChunkV1]:
    """Generate streaming response with native v1 chunks."""

    # Convert v1 messages to provider format
    provider_messages = [
        _convert_from_v1_to_provider_format(msg) for msg in messages
    ]

    # Stream from provider API
    for chunk in self._provider_stream(provider_messages, **kwargs):
        # Convert each chunk to v1 format
        v1_chunk = _convert_chunk_to_v1(chunk)
        yield v1_chunk

def _convert_chunk_to_v1(chunk: dict[str, Any]) -> AIMessageChunkV1:
    """Convert provider chunk to AIMessageChunkV1."""
    content: list[ContentBlock] = []

    if delta := chunk.get("delta"):
        if text := delta.get("content"):
            content.append(types.TextContentBlock(type="text", text=text))

        if tool_calls := delta.get("tool_calls"):
            for tool_call in tool_calls:
                if tool_call.get("id"):
                    content.append(types.ToolCallContentBlock(
                        type="tool_call",
                        id=tool_call["id"],
                        name=tool_call.get("function", {}).get("name", ""),
                        args=tool_call.get("function", {}).get("arguments", ""),
                    ))

    return AIMessageChunkV1(
        content=content,
        response_metadata=ResponseMetadata(
            model_name=chunk.get("model"),
        ),
    )

# Note: _convert_chunk_to_v1 does NOT handle summing - that's handled by AIMessageChunkV1.__add__ automatically
```

### Phase 4: Content Block Support

#### 4.1 Standard Content Block Types

Support all standard v1 content blocks as defined in `libs/core/langchain_core/messages/content_blocks.py`

#### 4.2 Provider-Specific Extensions

Use **PEP 728 TypedDict with Typed Extra Items** for provider-specific content within standard blocks:

```python
# Provider-specific fields within standard content blocks
# PEP 728 allows extra keys beyond the defined TypedDict structure
text_block_with_extras: TextContentBlock = {
    "type": "text",
    "text": "Hello world",
    "provider_confidence": 0.95,        # Extra field: provider-specific confidence score
    "provider_metadata": {              # Extra field: nested provider metadata
        "model_tier": "premium",
        "processing_time_ms": 150
    }
}
```

**About [PEP 728](https://peps.python.org/pep-0728/):** extends TypedDict to support typed extra items beyond the explicitly defined keys. This allows providers to add custom fields while maintaining type safety for the standard fields.

**Extra Item Types:** Standard content blocks support extra keys via PEP 728 with `extra_items=Any`, meaning provider-specific fields can be of any type. This provides:

- **Core fields** (like `type`, `text`, `id`) are **strongly typed** according to the TypedDict definition
- **Extra fields** (provider-specific extensions) are typed as `Any`, allowing complete flexibility
- **Type safety** is maintained for the standard fields while allowing arbitrary extensions

This is the most flexible approach - providers can add any kind of metadata, configuration, or custom data they need without breaking the type system or requiring changes to the core LangChain types.

```python
def _handle_provider_specific_content(block: dict[str, Any]) -> ContentBlock:
    """Handle provider-specific content blocks."""

    # For known provider extensions, create typed blocks
    if block.get("type") == "provider_specific_type":
        return cast(types.ContentBlock, ProviderSpecificContentBlock(...))

    # For unknown types, use NonStandardContentBlock
    return cast(types.ContentBlock, types.NonStandardContentBlock(
        type="non_standard",
        content=block
    ))
```

### Phase 5: Testing Framework

#### 5.1 V1 Tests

Create comprehensive tests for v1 functionality:

```python
from langchain_core.messages.v1 import HumanMessage as HumanMessageV1
from langchain_core.messages.content_blocks import TextContentBlock, ImageContentBlock, AudioContentBlock

def test_v1_native_message_handling():
    """Test native v1 message processing."""
    llm = ChatProviderV1(model="test-model")

    message = HumanMessageV1(content=[
        TextContentBlock(type="text", text="Hello"),
        ImageContentBlock(type="image", mime_type="image/jpeg", base64="base64data...")
    ])

    response = llm.invoke([message])

    assert isinstance(response, AIMessageV1)
    assert isinstance(response.content, list)
    assert all(isinstance(block, ContentBlock) for block in response.content)

    # Verify content block structure and content
    text_blocks = [b for b in response.content if b.get("type") == "text"]
    assert len(text_blocks) >= 1, "Response should contain at least one text block"
    assert text_blocks[0]["text"], "Text block should contain non-empty text content"
    assert isinstance(text_blocks[0]["text"], str), "Text content should be a string"

def test_v1_streaming_consistency():
    """Test that streaming and non-streaming produce equivalent results."""
    llm = ChatProviderV1(model="test-model")

    message = HumanMessageV1(content=[
        TextContentBlock(type="text", text="Hello"),
    ])

    # Non-streaming
    non_stream = llm.invoke([message])

    # Streaming
    stream_chunks = list(llm.stream([message]))
    stream_combined = AIMessageV1(content=[])
    for chunk in stream_chunks:
        stream_combined = stream_combined + chunk

    # Should be equivalent
    assert non_stream.content == stream_combined.content

def test_v1_content_block_types():
    """Test all supported content block types."""
    llm = ChatProviderV1(model="test-model")

    # Test each content block type
    test_cases = [
        TextContentBlock(type="text", text="Hello"),
        ImageContentBlock(type="image", mime_type="image/jpeg", base64="base64data..."),
        AudioContentBlock(type="audio", mime_type="audio/wav", base64="audiodata..."),
        # ...
    ]

    for block in test_cases:
        message = HumanMessageV1(content=[block])
        response = llm.invoke([message])
        assert isinstance(response, AIMessageV1)
```

#### 5.2 Migration Tests

Test compatibility between v0 and v1 implementations:

```python
from langchain_core.messages.content_blocks import TextContentBlock

def test_v0_v1_feature_parity():
    """Test that v1 implementation has feature parity with v0."""
    llm_v0 = ChatProvider(model="test-model")
    llm_v1 = ChatProviderV1(model="test-model")

    # Test basic functionality
    v0_response = llm_v0.invoke("Hello")
    v1_response = llm_v1.invoke([HumanMessageV1("Hello")
    ])
    v1_response = llm_v1.invoke([HumanMessageV1(content=[
        TextContentBlock(type="text", text="Hello")
    ])])

    # Extract text content for comparison
    v0_text = v0_response.content
    v1_text = "".join(
        block.get("text", "") for block in v1_response.content
        if block.get("type") == "text"
    )

    # Should produce equivalent text output
    assert v0_text == v1_text
```

### Phase 6: Documentation and Migration

#### 6.1 V1 Documentation

Document the v1 implementation separately:

```python
class ChatProviderV1(BaseChatProvider):
    """Provider chat model with native v1 content block support.

    This implementation provides native support for structured content blocks
    and always returns AIMessageV1 format responses.

    Examples:
        Basic text conversation:

        .. code-block:: python

            from langchain_{provider}.chat_models import ChatProviderV1
            from langchain_core.messages.v1 import HumanMessage

            llm = ChatProviderV1(model="provider-model")
            response = llm.invoke([
                HumanMessage(content=[
                    TextContentBlock(type="text", text="Hello!")
                ])
            ])

            # Response is always structured
            print(response.content)
            # [{"type": "text", "text": "Hello! How can I help?"}]  # Type will be TextContentBlock

        Multi-modal input:

        .. code-block:: python

            response = llm.invoke([
                HumanMessage(content=[
                    TextContentBlock(type="text", text="Describe this image:"),
                    ImageContentBlock(
                        type="image",
                        mime_type="image/jpeg",
                        base64="base64_encoded_image"
                    )
                ])
            ])
    """
```

#### 6.2 Migration Guide

```markdown
# Migrating to V1 Chat Models

## Overview

V1 chat models provide native support for standard content blocks and always return `AIMessageV1` format responses.

## Key Differences

### Import Changes
```python
# V0 implementation (conditional v1 support)
from langchain_{provider} import ChatProvider
llm = ChatProvider(output_version="v1")

# V1 implementation (v1 support)
from langchain_{provider}.chat_models import ChatProviderV1
llm = ChatProviderV1()
```

### Message Format

```python
# V0 mixed format (strings or lists)
message = HumanMessage(content="Hello")  # or content=[...]

# V1 structured format (always lists)
from langchain_core.messages.v1 import HumanMessage as HumanMessageV1
message = HumanMessageV1(content=[{"type": "text", "text": "Hello"}])
```

# Checklist

```txt
Core Infrastructure
- [ ] Create `base_v1.py` file with `BaseChatModelV1` inheritance
- [ ] Implement `ChatProviderV1` class with native v1 support
- [ ] Create `_compat.py` with v1 conversion utilities
- [ ] Update package `__init__.py` exports

Message Processing
- [ ] Implement `_convert_from_v1_to_provider_format()` for API requests
- [ ] Implement `_convert_to_v1_from_provider_format()` for responses
- [ ] Add streaming support with `AIMessageChunkV1`
- [ ] Handle all standard content block types

Content Block Support
- [ ] Support `TextContentBlock` for text content
- [ ] Support `ImageContentBlock` for images (where applicable)
- [ ] Support `AudioContentBlock` for audio (where applicable)
- [ ] Support `ToolCallContentBlock` for tool calls
- [ ] Support `ReasoningContentBlock` for reasoning (where applicable)
- [ ] Support other multimodal content blocks (where applicable)
- [ ] Handle provider-specific fields with `extra_items`
- [ ] Handle provider-specific blocks by returning `NonStandardContentBlock`

Testing
- [ ] Create comprehensive unit tests for v1 functionality
- [ ] Add streaming vs non-streaming consistency tests
- [ ] Test all supported content block types
- [ ] Add migration compatibility tests
- [ ] Performance benchmarks vs v0 implementation

Documentation
- [ ] Update class docstrings with v1 examples
- [ ] Create migration guide from v0 to v1
- [ ] Document provider-specific content block support
- [ ] Add troubleshooting section

## Success Criteria

1. **Native V1 Support**: Full `BaseChatModelV1` implementation
2. **Zero Conversion Overhead**: No runtime format conversion
3. **Feature Complete**: All provider capabilities available natively
4. **Type Safe**: Full typing for all content blocks and operations
5. **Well Documented**: Clear migration path and usage examples
6. **Performance Optimized**: Better performance than v0 conditional approach

## Provider-Specific Considerations

When implementing v1 support for your specific provider:

### Content Block Mapping
- Map your provider's native content types to standard v1 content blocks
- Use `NonStandardContentBlock` for provider-specific content that doesn't map to standard types
- Leverage PEP 728 TypedDict extra items for provider-specific fields/metadata within LangChain's standard blocks

### Tool Call Handling
- Map your provider's tool calling format to v1 `ToolCall` content blocks
- Handle both standard function calls and provider-specific built-in tools
- Preserve provider-specific tool call metadata using extra fields

### Streaming Implementation
- Ensure streaming chunks are properly typed as `AIMessageChunkV1`
- Implement proper chunk merging using the `+` operator
- Handle provider-specific streaming features (like reasoning) appropriately
```

## Common Implementation Mistakes & Solutions

### Message Structure Assumptions

**❌ Mistake**: Assuming `ChatMessage` exists in v1 messages

```python
from langchain_core.messages.v1 import ChatMessage as ChatMessageV1  # DOES NOT EXIST
```

**✅ Solution**: V1 only has `AIMessage`, `HumanMessage`, `SystemMessage`, `ToolMessage`, `AIMessageChunk`

```python
from langchain_core.messages.v1 import (
    AIMessage as AIMessageV1,
    HumanMessage as HumanMessageV1,
    SystemMessage as SystemMessageV1,
    ToolMessage as ToolMessageV1,
    AIMessageChunk as AIMessageChunkV1,
    MessageV1,  # Union type for all message types
)
```

### Reasoning Content Handling

**❌ Mistake**: Storing reasoning content in `additional_kwargs` or `response_metadata`

V1 messages don't have `additional_kwargs`

**✅ Solution**: Use `ReasoningContentBlock` in the content list

```python
from langchain_core.messages.content_blocks import ReasoningContentBlock

if reasoning_content:
    content.append(ReasoningContentBlock(
        type="reasoning",
        reasoning=reasoning_content,
    ))
```

Any other reasoning-related metadata should be stored as extra items in the content block. For instance, thought signatures or effort.

### ResponseMetadata Type Safety

**❌ Mistake**: Directly assigning provider-specific fields to ResponseMetadata

```python
response_metadata["created_at"] = response["created_at"]  # Type error
```

**✅ Solution**: Use type ignores for extra fields (allowed by PEP 728)

```python
response_metadata["created_at"] = response["created_at"]  # type: ignore[typeddict-unknown-key]
```

### Streaming Implementation

**❌ Mistake**: Returning wrong message type from streaming methods

```python
def _generate_stream(self) -> Iterator[AIMessageChunkV1]:
    # ...
    ai_message = _convert_to_v1_from_ollama_format(response)  # Returns AIMessageV1
    yield ai_message  # Type error: expected AIMessageChunkV1
```

**✅ Solution**: Convert AIMessageV1 to AIMessageChunkV1 for streaming

```python
ai_message = _convert_to_v1_from_ollama_format(response)
chunk = AIMessageChunkV1(
    content=ai_message.content,
    response_metadata=ai_message.response_metadata,
    usage_metadata=ai_message.usage_metadata,
)
yield chunk
```

### Generator Type Issues

**❌ Mistake**: Using `.get()` on ContentBlock without type conversion

**Why is this a problem?** While ContentBlock is a Union of TypedDict classes (like `TextContentBlock`, `ToolCall`, etc.), mypy treats accessing fields on the union as returning `Any` type because it can't guarantee which specific TypedDict variant you have. When you pass this to `"".join()`, mypy expects `Iterable[str]` but gets `Iterable[Any]`, causing a type error.

```python
text_content = "".join(
    block.get("text", "")  # Returns Any, treated as object in strict mode
    for block in chunk.content
    if block.get("type") == "text"
)
```

**✅ Solution**: Explicitly convert to string

```python
text_content = "".join(
    str(block.get("text", ""))
    for block in chunk.content
    if block.get("type") == "text"
)
```

### Test Type Annotations

**❌ Mistake**: Missing return type annotations in tests

```python
def test_something(self):  # mypy error: no return type
    pass
```

**✅ Solution**: Add `-> None` to all test methods

```python
def test_something(self) -> None:
    pass
```

### Performance Optimizations

**❌ Mistake**: Using append() in loops for list construction

```python
for tool_call in tool_calls:
    content.append(ToolCall(...))  # Flagged by PERF401
```

**✅ Solution**: Use list comprehension with extend()

```python
content.extend([
    ToolCall(
        type="tool_call",
        id=tool_call.get("id", str(uuid4())),
        name=tool_call["function"]["name"],
        args=tool_call["function"]["arguments"],
    )
    for tool_call in tool_calls
])
```

## Implementation Debugging Tips

### Type Checking Issues

1. Run `mypy` early and often - type issues compound quickly
2. Use `# type: ignore[specific-error]` comments for intentional violations
3. Check `ResponseMetadata` usage - it allows extra keys but mypy doesn't know this

### Import Issues

1. V1 message imports are different - no `ChatMessage`, limited to core message types
2. Always use grouped imports from the same module to avoid linter issues
3. Remove unused imports that can be auto-detected

### Testing Strategy

1. Test message conversion utilities separately from the main chat model
2. Use concrete test data rather than mocking for content block testing
3. Include both streaming and non-streaming test cases
4. Test error conditions (empty content, missing fields, etc.)

### Common Error Patterns

- **"AIMessage has no attribute 'additional_kwargs'"** → Use ReasoningContentBlock instead
- **"ChatMessage not found"** → ChatMessage doesn't exist in V1, remove references
- **Type errors with ResponseMetadata** → Use `.get()` instead of checking directly
  - Bad: `assert result.response_metadata["model_name"] == MODEL_NAME  # type: ignore[typeddict-not-required-key]`
  - Good: `assert result.response_metadata.get("model_name") == MODEL_NAME`

- **Generator type mismatches** → Use `str()` conversion for text extraction from content blocks
