"""Standard integration tests for `ChatOpenRouter`."""

from typing import Literal

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_openrouter.chat_models import ChatOpenRouter

MODEL_NAME = "openai/gpt-4o-mini"


class TestChatOpenRouter(ChatModelIntegrationTests):
    """Test `ChatOpenRouter` chat model."""

    @property
    def chat_model_class(self) -> type[ChatOpenRouter]:
        """Return class of chat model being tested."""
        return ChatOpenRouter

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "temperature": 0,
        }

    @property
    def supports_json_mode(self) -> bool:
        return False

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True

    @property
    def supports_audio_inputs(self) -> bool:
        return True

    @property
    def supports_video_inputs(self) -> bool:
        return True

    @property
    def supports_pdf_inputs(self) -> bool:
        return True

    @property
    def model_override_value(self) -> str:
        return "openai/gpt-4o"

    @property
    def supported_usage_metadata_details(
        self,
    ) -> dict[
        Literal["invoke", "stream"],
        list[
            Literal[
                "audio_input",
                "audio_output",
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ]
        ],
    ]:
        return {
            "invoke": [
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ],
            "stream": [
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ],
        }
