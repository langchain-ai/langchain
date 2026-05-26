"""Unit tests for `ChatAtlas`."""

from __future__ import annotations

import pytest
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import SecretStr

from langchain_atlas.chat_models import DEFAULT_API_BASE, ChatAtlas

MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"


class TestChatAtlasUnit(ChatModelUnitTests):
    """Standard unit tests for `ChatAtlas` chat model."""

    @property
    def chat_model_class(self) -> type[ChatAtlas]:
        """Chat model class being tested."""
        return ChatAtlas

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters to initialize from environment variables."""
        return (
            {
                "ATLAS_API_KEY": "api_key",
                "ATLAS_API_BASE": "api_base",
            },
            {
                "model": MODEL_NAME,
            },
            {
                "atlas_api_key": "api_key",
                "atlas_api_base": "api_base",
            },
        )

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "api_key": "api_key",
        }


class TestChatAtlasCustomUnit:
    """Custom tests specific to Atlas Cloud chat model."""

    def test_base_url_alias(self) -> None:
        """Test that `base_url` is accepted as an alias."""
        chat_model = ChatAtlas(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
            base_url="http://example.test/v1",
        )
        assert chat_model.atlas_api_base == "http://example.test/v1"

    def test_default_base_url(self) -> None:
        """Test that the default Atlas base URL is set."""
        chat_model = ChatAtlas(model=MODEL_NAME, api_key=SecretStr("api_key"))
        assert chat_model.atlas_api_base == DEFAULT_API_BASE

    def test_llm_type(self) -> None:
        """Test that `_llm_type` identifies Atlas."""
        chat_model = ChatAtlas(model=MODEL_NAME, api_key=SecretStr("api_key"))
        assert chat_model._llm_type == "atlas-chat"

    def test_ls_provider(self) -> None:
        """Test that LangSmith provider is Atlas."""
        chat_model = ChatAtlas(model=MODEL_NAME, api_key=SecretStr("api_key"))
        assert chat_model._get_ls_params()["ls_provider"] == "atlas"

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment variable fallback for API key."""
        monkeypatch.setenv("ATLAS_API_KEY", "env-key")
        chat_model = ChatAtlas(model=MODEL_NAME)
        assert chat_model.atlas_api_key is not None
        assert chat_model.atlas_api_key.get_secret_value() == "env-key"

    def test_api_base_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment variable fallback for base URL."""
        monkeypatch.setenv("ATLAS_API_BASE", "https://env.example.test/v1")
        chat_model = ChatAtlas(model=MODEL_NAME, api_key=SecretStr("api_key"))
        assert chat_model.atlas_api_base == "https://env.example.test/v1"

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing API key raises ValueError."""
        monkeypatch.delenv("ATLAS_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Atlas API key is not set"):
            ChatAtlas(model=MODEL_NAME)

    def test_invalid_streaming_params(self) -> None:
        """Test that invalid `n` with streaming raises ValueError."""
        with pytest.raises(ValueError, match="n must be 1 when streaming"):
            ChatAtlas(
                model=MODEL_NAME,
                api_key=SecretStr("api_key"),
                streaming=True,
                n=2,
            )
