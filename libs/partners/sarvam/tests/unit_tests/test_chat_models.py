"""Unit tests for ChatSarvam."""

from typing import Any

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_sarvam.chat_models import ChatSarvam


class TestChatSarvamUnit(ChatModelUnitTests):
    """Unit tests for ChatSarvam."""

    @property
    def chat_model_class(self) -> type[ChatSarvam]:
        """Get chat model class."""
        return ChatSarvam

    @property
    def chat_model_params(self) -> dict[str, Any]:
        """Get chat model parameters."""
        return {
            "model": "sarvam-1",
            "api_key": "test-api-key",
            "temperature": 0.7,
        }


def test_initialization() -> None:
    """Test ChatSarvam initialization."""
    llm = ChatSarvam(model="sarvam-1", api_key="test-key")
    assert llm.model == "sarvam-1"
    assert llm.temperature == 0.7


def test_sarvam_model_param() -> None:
    """Test model parameter."""
    llm = ChatSarvam(model="sarvam-2", api_key="test-key")
    assert llm.model == "sarvam-2"


def test_sarvam_model_kwargs() -> None:
    """Test model kwargs."""
    llm = ChatSarvam(
        model="sarvam-1",
        api_key="test-key",
        model_kwargs={"custom_param": "value"},
    )
    params = llm._identifying_params
    assert params["custom_param"] == "value"
