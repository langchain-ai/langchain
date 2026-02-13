"""Standard unit tests for `ChatOpenRouter`."""

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_openrouter.chat_models import ChatOpenRouter

MODEL_NAME = "openai/gpt-4o-mini"


class TestChatOpenRouterUnit(ChatModelUnitTests):
    """Standard unit tests for `ChatOpenRouter` chat model."""

    @property
    def chat_model_class(self) -> type[ChatOpenRouter]:
        """Chat model class being tested."""
        return ChatOpenRouter

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters to initialize from environment variables."""
        return (
            {
                "OPENROUTER_API_KEY": "api_key",
            },
            {
                "model": MODEL_NAME,
            },
            {
                "openrouter_api_key": "api_key",
            },
        )

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "api_key": "test-api-key",
        }
