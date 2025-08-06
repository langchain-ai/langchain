# Standard Tests V1 - Content Blocks Support

## Overview

The standard tests v1 package provides comprehensive testing for chat models that support the new content blocks format. This includes:

- **Streaming support**: Content blocks in streaming responses
- **Multimodal content**: `Text`, `Image`, `Video`, `Audio`, and `File` `ContentBlock`s
- **Reasoning content**: Reasoning steps as `ReasoningContentBlock`
- **Provider-specific extensions**: `NonStandardContentBlock` for unique provider features

## Usage

### Basic Unit Tests

```python
from langchain_tests.unit_tests.chat_models_v1 import ChatModelV1UnitTests
from your_package import YourChatModel

class TestYourChatModelV1(ChatModelV1UnitTests):
    @property
    def chat_model_class(self):
        return YourChatModel

    @property
    def chat_model_params(self):
        return {"api_key": "test-key", "model": "your-model"}

    # Configure supported features
    @property
    def supports_content_blocks_v1(self):
        return True

    @property
    def supports_image_content_blocks(self):
        return True

    @property
    def supports_reasoning_content_blocks(self):
        return True
```

### Integration Tests

```python
from langchain_tests.integration_tests.chat_models_v1 import ChatModelV1IntegrationTests
from your_package import YourChatModel

class TestYourChatModelV1Integration(ChatModelV1IntegrationTests):
    @property
    def chat_model_class(self):
        return YourChatModel

    @property
    def chat_model_params(self):
        return {
            "api_key": os.getenv("YOUR_API_KEY"),
            "model": "your-model-name"
        }

    # Configure which features to test
    @property
    def supports_citations(self):
        return True

    @property
    def supports_web_search_blocks(self):
        return False  # If your model doesn't support this
```

## Configuration Properties

### Core Content Blocks Support

- `supports_content_blocks_v1`: Enable content blocks v1 testing **(required)**
- `supports_text_content_blocks`: `TextContentBlock` support - very unlikely this will be set to `False`
- `supports_reasoning_content_blocks`: `ReasoningContentBlock` support, e.g. for reasoning models

### Multimodal Support

- `supports_image_content_blocks`: `ImageContentBlock`s (v1 format)
- `supports_video_content_blocks`: `VideoContentBlock`s (v1 format)
- `supports_audio_content_blocks`: `AudioContentBlock`s (v1 format)
- `supports_plaintext_content_blocks`: `PlainTextContentBlock`s (plaintext from documents)
- `supports_file_content_blocks`: `FileContentBlock`s

### Tool Calling

- `supports_tool_calls`: Tool calling with content blocks
- `supports_invalid_tool_calls`: Error handling for invalid tool calls
- `supports_tool_call_chunks`: Streaming tool call support

### Advanced Features

- `supports_citations`: Citation annotations
- `supports_web_search_blocks`: Built-in web search
- `supports_code_interpreter`: Code execution blocks
- `supports_non_standard_blocks`: Custom content blocks

## Test Categories

### Unit Tests (`ChatModelV1Tests`)

- Content block format validation
- Ser/deserialization
- Multimodal content handling
- Tool calling with content blocks
- Error handling for invalid blocks
- Backward compatibility with string content

### Integration Tests (`ChatModelV1IntegrationTests`)

- Real multimodal content processing
- Advanced reasoning with content blocks
- Citation generation with external sources
- Web search integration
- File processing and analysis
- Performance benchmarking
- Streaming content blocks
- Asynchronous processing

## Migration from Standard Tests

### For Test Authors

1. **Inherit from new base classes**:

   ```python
   # v0
   from langchain_tests.unit_tests.chat_models import ChatModelUnitTests

   # v1
   from langchain_tests.unit_tests.chat_models_v1 import ChatModelV1UnitTests
   ```

2. **Configure content blocks support**:

   ```python
   @property
   def supports_content_blocks_v1(self):
       return True  # Enable v1 features
   ```

3. **Set feature flags** based on your chat model's capabilities

## Examples

See the test files in `tests/unit_tests/test_chat_models_v1.py` and `tests/integration_tests/test_chat_models_v1.py` for complete examples of how to implement tests for your chat model.

## Best Practices

1. **Start with basic content blocks** (text) and gradually enable advanced features
2. **Test error handling** for unsupported content block types
3. **Validate serialization** to persist message histories (passing back in content blocks)
4. **Test streaming** if your model supports it with content blocks

## Contributing

When new content block types or features are added:

1. Add the content block type to the imports
2. Create test helper methods for the new type
3. Add configuration properties for the feature
4. Implement corresponding test methods
5. Update this documentation
6. Add examples in the test files (`tests/unit_tests/test_chat_models_v1.py` and `tests/integration_tests/test_chat_models_v1.py`)
