# Standard Tests V1 - Quick Start Guide

This guide shows you how to quickly get started with the new content blocks v1 test suite.

## 🚀 Quick Usage

### 1. Basic Setup

New imports:

```python
# v0
from langchain_tests.unit_tests.chat_models import ChatModelUnitTests

# v1
from langchain_tests.v1.unit_tests.chat_models import ChatModelUnitTests as ChatModelV1UnitTests
```

### 2. Minimal Configuration

```python
class TestMyChatModelV1(ChatModelV1UnitTests):
    @property
    def chat_model_class(self):
        return MyChatModelV1

    # Enable content blocks support
    @property
    def supports_content_blocks_v1(self):
        return True

    # The rest should be the same
    @property
    def chat_model_params(self):
        return {"api_key": "test-key"}
```

### 3. Run Tests

```bash
uv run --group test pytest tests/unit_tests/test_my_model_v1.py -v
```

## ⚙️ Feature Configuration

Like before, only enable the features your model supports:

```python
class TestAdvancedModelV1(ChatModelV1UnitTests):
    # REQUIRED
    @property
    def supports_content_blocks_v1(self):
        return True

    # Multimodal features
    @property
    def supports_image_content_blocks(self):
        return True  # ✅ Enable if supported

    @property
    def supports_video_content_blocks(self):
        return False  # ❌ Disable if not supported, but will default to False if not explicitly set

    # Advanced features
    @property
    def supports_reasoning_content_blocks(self):
        """Model generates reasoning steps"""
        return True

    @property
    def supports_citations(self):
        """Model provides source citations"""
        return True

```

## 📋 Feature Reference

| Property | Description | Default |
|----------|-------------|---------|
| `supports_content_blocks_v1` | Core content blocks support | `True` |
| `supports_text_content_blocks` | Basic text blocks | `True` |
| `supports_image_content_blocks` | Image content blocks (v1) | `False` |
| `supports_video_content_blocks` | Video content blocks (v1) | `False` |
| `supports_audio_content_blocks` | Audio content blocks (v1) | `False` |
| `supports_file_content_blocks` | File content blocks | `False` |
| `supports_reasoning_content_blocks` | Reasoning/thinking blocks | `False` |
| `supports_citations` | Citation annotations | `False` |
| `supports_web_search_blocks` | Web search integration | `False` |
| `supports_enhanced_tool_calls` | Tool calling | `False` |
| `supports_non_standard_blocks` | Custom content blocks | `True` |

**Note:** These defaults are provided by the base test class. You only need to override properties where your model's capabilities differ from the default.

## 🔧 Common Patterns

### For Text-Only Models

```python
@property
def supports_content_blocks_v1(self):
    return True

# All multimodal features inherit False defaults from base class
# No need to override them unless your model supports them
```

### For Multimodal Models

Set the v1 content block features that your model supports:

- `supports_image_content_blocks`
- `supports_video_content_blocks`
- `supports_audio_content_blocks`

### For Advanced AI Models

Set the features that your model supports, including reasoning and citations:

- `supports_reasoning_content_blocks`
- `supports_citations`
- `supports_web_search_blocks`

## 🚨 Troubleshooting

### Tests Failing?

1. **Check feature flags** - Only enable what your model actually supports
2. **Verify API keys** - Integration tests may need credentials
3. **Check model parameters** - Make sure initialization params are correct

### Tests Skipping?

This is normal! Tests skip automatically when features aren't supported. Only tests for enabled features will run.

## 🏃‍♂️ Migration Checklist

- [ ] Update test base class imports
- [ ] Add `supports_content_blocks_v1 = True`
- [ ] Configure feature flags based on model capabilities
- [ ] Run tests to verify configuration
- [ ] Adjust any failing/skipping tests as needed

## 📚 Next Steps

- Read `README_V1.md` for complete feature documentation
- Look at `tests/unit_tests/test_chat_models_v1.py` for working examples

# Example Files

## Unit Tests

```python
"""Example test implementation using ``ChatModelV1UnitTests``.

This file demonstrates how to use the new content blocks v1 test suite
for testing chat models that support the enhanced content blocks system.
"""

from typing import Any

from langchain_core.v1.language_models.chat_models import BaseChatModelV1
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages.content_blocks import TextContentBlock

from langchain_tests.unit_tests.chat_models_v1 import ChatModelV1UnitTests


class FakeChatModelV1(GenericFakeChatModel):
    """Fake chat model that supports content blocks v1 format.

    This is a test implementation that demonstrates content blocks support.
    """

    def _call(self, messages: Any, stop: Any = None, **kwargs: Any) -> BaseMessage:
        """Override to handle content blocks format."""
        # Process messages and handle content blocks
        response = super()._call(messages, stop, **kwargs)

        # Convert response to content blocks format if needed
        if isinstance(response.content, str):
            # Convert string response to TextContentBlock format
            from langchain_core.messages import AIMessage

            text_block: TextContentBlock = {"type": "text", "text": response.content}
            return AIMessage(content=[text_block])

        return response


class TestFakeChatModelV1(ChatModelV1UnitTests):
    """Test implementation using the new content blocks v1 test suite."""

    @property
    def chat_model_class(self) -> type[BaseChatModelV1]:
        """Return the fake chat model class for testing."""
        return FakeChatModelV1

    @property
    def chat_model_params(self) -> dict[str, Any]:
        """Parameters for initializing the fake chat model."""
        return {
            "messages": iter(
                [
                    "This is a test response with content blocks support.",
                    "Another test response for validation.",
                    "Final test response for comprehensive testing.",
                ]
            )
        }

    # Content blocks v1 support configuration
    @property
    def supports_content_blocks_v1(self) -> bool:
        """This fake model supports content blocks v1."""
        return True

    @property
    def supports_text_content_blocks(self) -> bool:
        """This fake model supports TextContentBlock."""
        return True

    @property
    def supports_reasoning_content_blocks(self) -> bool:
        """This fake model does not support ReasoningContentBlock."""
        return False

    @property
    def supports_citations(self) -> bool:
        """This fake model does not support citations."""
        return False

    @property
    def supports_tool_calls(self) -> bool:
        """This fake model supports tool calls."""
        return True

    @property
    def has_tool_calling(self) -> bool:
        """Enable tool calling tests."""
        return True

    @property
    def supports_image_content_blocks(self) -> bool:
        """This fake model does not support image content blocks."""
        return False

    @property
    def supports_non_standard_blocks(self) -> bool:
        """This fake model supports non-standard blocks."""
        return True
```

## Integration Tests

```python
"""Example integration test implementation using ChatModelV1IntegrationTests.

This file demonstrates how to use the new content blocks v1 integration test suite
for testing real chat models that support the enhanced content blocks system.

.. note::
  This is a template/example. Real implementations should replace ``FakeChatModelV1``
  with actual chat model classes.

"""

import os
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel, GenericFakeChatModel

from langchain_tests.v1.integration_tests.chat_models import ChatModelIntegrationTests as ChatModelV1IntegrationTests


# Example fake model for demonstration (replace with real model in practice)
class FakeChatModelV1Integration(GenericFakeChatModel):
    """Fake chat model for integration testing demonstration."""

    @property
    def _llm_type(self) -> str:
        return "fake_chat_model_v1_integration"


class TestFakeChatModelV1Integration(ChatModelV1IntegrationTests):
    """Example integration test using content blocks v1 test suite.

    In practice, this would test a real chat model that supports content blocks.
    Replace FakeChatModelV1Integration with your actual chat model class.
    """

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        """Return the chat model class to test."""
        return FakeChatModelV1Integration

    @property
    def chat_model_params(self) -> dict[str, Any]:
        """Parameters for initializing the chat model."""
        return {
            "messages": iter(
                [
                    "Integration test response with content blocks.",
                    "Multimodal content analysis response.",
                    "Tool calling response with structured output.",
                    "Citation-enhanced response with sources.",
                    "Web search integration response.",
                ]
            )
        }

    # Content blocks v1 support configuration
    @property
    def supports_content_blocks_v1(self) -> bool:
        """Enable content blocks v1 testing."""
        return True

    @property
    def supports_text_content_blocks(self) -> bool:
        """Enable TextContentBlock testing."""
        return True

    @property
    def supports_reasoning_content_blocks(self) -> bool:
        """Disable reasoning blocks for this fake model."""
        return False

    @property
    def supports_citations(self) -> bool:
        """Disable citations for this fake model."""
        return False

    @property
    def supports_web_search_blocks(self) -> bool:
        """Disable web search for this fake model."""
        return False

    @property
    def has_tool_calling(self) -> bool:
        """Enable tool calling tests."""
        return True

    @property
    def supports_image_inputs(self) -> bool:
        """Disable image inputs for this fake model."""
        return False

    @property
    def supports_video_inputs(self) -> bool:
        """Disable video inputs for this fake model."""
        return False

    @property
    def supports_audio_inputs(self) -> bool:
        """Disable audio inputs for this fake model."""
        return False

    @property
    def supports_file_content_blocks(self) -> bool:
        """Disable file content blocks for this fake model."""
        return False

    @property
    def supports_non_standard_blocks(self) -> bool:
        """Enable non-standard blocks support."""
        return True


# Example of a more realistic integration test configuration
# that would require API keys and external services
class TestRealChatModelV1IntegrationTemplate(ChatModelV1IntegrationTests):
    """Template for testing real chat models with content blocks v1.

    This class shows how you would configure tests for a real model
    that requires API keys and supports various content block features.
    """

    @pytest.fixture(scope="class", autouse=True)
    def check_api_key(self) -> None:
        """Check that required API key is available."""
        if not os.getenv("YOUR_MODEL_API_KEY"):
            pytest.skip("YOUR_MODEL_API_KEY not set, skipping integration tests")

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        """Return your actual chat model class."""
        # Replace with your actual model, e.g.:
        # from your_package import YourChatModel
        # return YourChatModel
        return FakeChatModelV1Integration  # Placeholder

    @property
    def chat_model_params(self) -> dict[str, Any]:
        """Parameters for your actual chat model."""
        return {
            # "api_key": os.getenv("YOUR_MODEL_API_KEY"),
            # "model": "your-model-name",
            # "temperature": 0.1,
            # Add your model's specific parameters
        }

    # Configure which features your model supports
    @property
    def supports_content_blocks_v1(self) -> bool:
        return True  # Set based on your model's capabilities

    @property
    def supports_image_inputs(self) -> bool:
        return True  # Set based on your model's capabilities

    @property
    def supports_reasoning_content_blocks(self) -> bool:
        return True  # Set based on your model's capabilities

    @property
    def supports_citations(self) -> bool:
        return True  # Set based on your model's capabilities

    @property
    def supports_web_search_blocks(self) -> bool:
        return False  # Set based on your model's capabilities

    @property
    def supports_enhanced_tool_calls(self) -> bool:
        return True  # Set based on your model's capabilities

    @property
    def has_tool_calling(self) -> bool:
        return True  # Set based on your model's capabilities

    # Add any model-specific test overrides or skips
    @pytest.mark.skip(reason="Template class - not for actual testing")
    def test_all_inherited_tests(self) -> None:
        """This template class should not run actual tests."""
        pass

```
