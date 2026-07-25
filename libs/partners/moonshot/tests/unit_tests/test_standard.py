"""Standard unit tests for `ChatMoonshot`."""

from __future__ import annotations

from typing import ClassVar

from langchain_tests.unit_tests.chat_models import ChatModelUnitTests

from langchain_moonshot.chat_models import ChatMoonshot


class TestMoonshotStandard(ChatModelUnitTests):
    """Standard ChatModel unit tests for Moonshot."""

    @property
    def chat_model_class(self) -> type[ChatMoonshot]:
        return ChatMoonshot

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "moonshot-v1-8k",
            "api_key": "test-api-key",
        }

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "MOONSHOT_API_KEY": "test-api-key",
            },
            {
                "model": "moonshot-v1-8k",
            },
            {
                "api_key": "test-api-key",
            },
        )

    @property
    def chat_model_has_tool_calling(self) -> bool:
        return True

    @property
    def chat_model_has_structured_output(self) -> bool:
        return True

    @property
    def chat_model_has_streaming(self) -> bool:
        return True

    @property
    def supports_image_inputs(self) -> bool:
        return False  # Only 128k model supports images

    @property
    def supports_video_inputs(self) -> bool:
        return False

    @property
    def supports_audio_inputs(self) -> bool:
        return False
