"""Standard integration tests for `ChatOpenRouter`."""

from typing import Literal

from langchain_core.messages import AIMessage, AIMessageChunk
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
    def supports_video_inputs(self) -> bool:
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
                "cache_read_input",
                "cache_creation_input",
            ],
            "stream": [
                "cache_read_input",
                "cache_creation_input",
            ],
        }


AUDIO_MODEL = "google/gemini-2.5-flash"
REASONING_MODEL = "openai/o3-mini"


class TestChatOpenRouterMultiModal(ChatModelIntegrationTests):
    """Tests for audio input and reasoning output capabilities.

    Uses an audio-capable model as the base and creates separate model
    instances for reasoning tests.
    """

    @property
    def chat_model_class(self) -> type[ChatOpenRouter]:
        return ChatOpenRouter

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": AUDIO_MODEL,
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

    def invoke_with_reasoning_output(self, *, stream: bool = False) -> AIMessage:
        """Invoke a reasoning model to exercise reasoning token tracking."""
        llm = ChatOpenRouter(
            model=REASONING_MODEL,
            reasoning={"effort": "medium"},
        )
        prompt = (
            "Explain the relationship between the 2008/9 economic crisis and "
            "the startup ecosystem in the early 2010s"
        )
        if stream:
            full: AIMessageChunk | None = None
            for chunk in llm.stream(prompt):
                full = chunk if full is None else full + chunk  # type: ignore[assignment]
            assert full is not None
            return full
        return llm.invoke(prompt)
