"""Test the standard v1 tests on the ``ChatParrotLinkV1`` custom chat model."""

import pytest

from langchain_tests.v1.unit_tests.chat_models import ChatModelUnitTests

from .custom_chat_model_v1 import ChatParrotLinkV1


class TestChatParrotLinkV1Unit(ChatModelUnitTests):
    """Unit tests for ``ChatParrotLinkV1`` using the standard v1 test suite."""

    @property
    def chat_model_class(self) -> type[ChatParrotLinkV1]:
        """Return the chat model class to test."""
        return ChatParrotLinkV1

    @property
    def chat_model_params(self) -> dict:
        """Return the parameters for initializing the chat model."""
        return {
            "model": "parrot-v1-test",
            "parrot_buffer_length": 20,
            "temperature": 0.0,
        }

    @pytest.fixture
    def model(self) -> ChatParrotLinkV1:
        """Create a model instance for testing."""
        return self.chat_model_class(**self.chat_model_params)

    # Override property methods to match ChatParrotLinkV1 capabilities
    @property
    def has_tool_calling(self) -> bool:
        """``ChatParrotLinkV1`` does not support tool calling."""
        return False

    @property
    def has_structured_output(self) -> bool:
        """``ChatParrotLinkV1`` does not support structured output."""
        return False

    @property
    def supports_json_mode(self) -> bool:
        """``ChatParrotLinkV1`` does not support JSON mode."""
        return False

    @property
    def supports_content_blocks_v1(self) -> bool:
        """``ChatParrotLinkV1`` supports content blocks v1 format."""
        return True

    @property
    def supports_text_content_blocks(self) -> bool:
        """``ChatParrotLinkV1`` supports ``TextContentBlock``."""
        return True

    @property
    def supports_non_standard_blocks(self) -> bool:
        """``ChatParrotLinkV1`` can handle ``NonStandardContentBlock`` gracefully."""
        return True

    # All other content block types are not supported by ChatParrotLinkV1
    @property
    def supports_reasoning_content_blocks(self) -> bool:
        """``ChatParrotLinkV1`` does not generate ``ReasoningContentBlock``."""
        return False

    @property
    def supports_image_content_blocks(self) -> bool:
        """``ChatParrotLinkV1`` does not support ``ImageContentBlock``."""
        return False

    @property
    def supports_audio_content_blocks(self) -> bool:
        """``ChatParrotLinkV1`` does not support ``AudioContentBlock``."""
        return False

    @property
    def supports_video_content_blocks(self) -> bool:
        """``ChatParrotLinkV1`` does not support ``VideoContentBlock``."""
        return False

    @property
    def supports_citations(self) -> bool:
        """``ChatParrotLinkV1`` does not support citations."""
        return False

    @property
    def supports_web_search_blocks(self) -> bool:
        """``ChatParrotLinkV1`` does not support web search blocks."""
        return False
