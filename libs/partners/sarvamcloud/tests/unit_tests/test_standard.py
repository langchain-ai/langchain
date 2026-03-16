"""Standard LangChain compliance tests for ChatSarvam."""

from typing import TYPE_CHECKING, Type
from unittest.mock import MagicMock, patch

import pytest
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_sarvamcloud.chat_models import ChatSarvam

if TYPE_CHECKING:
    pass


class TestChatSarvamStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatSarvam]:
        return ChatSarvam

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "sarvam-105b",
            "api_subscription_key": "test-key",
        }

    @pytest.fixture(autouse=True)
    def mock_sarvam_client(self) -> None:
        """Mock the sarvamai SDK for all standard tests."""
        with patch("sarvamai.SarvamAI"), patch("sarvamai.AsyncSarvamAI"):
            yield
