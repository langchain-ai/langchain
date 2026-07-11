"""Test chat model integration."""

from __future__ import annotations

from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import SecretStr

from langchain_requesty.chat_models import DEFAULT_API_BASE, ChatRequesty

MODEL_NAME = "openai/gpt-4o-mini"


class TestChatRequestyUnit(ChatModelUnitTests):
    """Standard unit tests for `ChatRequesty` chat model."""

    @property
    def chat_model_class(self) -> type[ChatRequesty]:
        """Chat model class being tested."""
        return ChatRequesty

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters to initialize from environment variables."""
        return (
            {
                "REQUESTY_API_KEY": "api_key",
                "REQUESTY_API_BASE": "api_base",
            },
            {
                "model": MODEL_NAME,
            },
            {
                "api_key": "api_key",
                "api_base": "api_base",
            },
        )

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "api_key": "api_key",
        }

    def get_chat_model(self) -> ChatRequesty:
        """Get a chat model instance for testing."""
        return ChatRequesty(**self.chat_model_params)


class TestChatRequestyCustomUnit:
    """Custom tests specific to Requesty chat model."""

    def test_default_api_base(self) -> None:
        """Test that the default base URL points at the Requesty router."""
        chat_model = ChatRequesty(model=MODEL_NAME, api_key=SecretStr("api_key"))
        assert chat_model.api_base == DEFAULT_API_BASE

    def test_base_url_alias(self) -> None:
        """Test that `base_url` is accepted as an alias for `api_base`."""
        chat_model = ChatRequesty(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
            base_url="http://example.test/v1",
        )
        assert chat_model.api_base == "http://example.test/v1"

    def test_llm_type(self) -> None:
        """Test that the model reports the expected `_llm_type`."""
        chat_model = ChatRequesty(model=MODEL_NAME, api_key=SecretStr("api_key"))
        assert chat_model._llm_type == "chat-requesty"

    def test_ls_params_provider(self) -> None:
        """Test that LangSmith params report the requesty provider."""
        chat_model = ChatRequesty(model=MODEL_NAME, api_key=SecretStr("api_key"))
        ls_params = chat_model._get_ls_params()
        assert ls_params.get("ls_provider") == "requesty"

    def test_lc_secrets(self) -> None:
        """Test that the API key is registered as a secret."""
        chat_model = ChatRequesty(model=MODEL_NAME, api_key=SecretStr("api_key"))
        assert chat_model.lc_secrets == {"api_key": "REQUESTY_API_KEY"}

    def test_attribution_headers(self) -> None:
        """Test that `app_url` and `app_title` map to attribution headers."""
        chat_model = ChatRequesty(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
            app_url="https://example.test",
            app_title="Example",
        )
        default_headers = chat_model.root_client.default_headers
        assert default_headers.get("HTTP-Referer") == "https://example.test"
        assert default_headers.get("X-Title") == "Example"

    def test_metadata_versions(self) -> None:
        """Test that metadata reports the correct version info."""
        chat_model = ChatRequesty(model=MODEL_NAME, api_key=SecretStr("api_key"))
        assert chat_model.metadata is not None
        versions = chat_model.metadata["lc_versions"]
        assert "langchain-core" in versions
        assert "langchain-requesty" in versions
        assert "langchain-openai" in versions
