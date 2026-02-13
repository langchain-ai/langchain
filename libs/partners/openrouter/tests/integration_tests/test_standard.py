"""Standard integration tests for `ChatOpenRouter`."""

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
        """Whether the chat model supports JSON mode."""
        return True

    @property
    def supports_image_inputs(self) -> bool:
        """Whether the chat model supports image inputs."""
        return True

    @property
    def supports_image_urls(self) -> bool:
        """Whether the chat model supports image inputs from URLs."""
        return True
