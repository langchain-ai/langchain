"""Test ChatRequesty chat model.

These tests hit the live Requesty router and require a valid `REQUESTY_API_KEY`
in the environment. They are skipped automatically when the key is not set.
"""

from __future__ import annotations

import os

import pytest
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_requesty.chat_models import ChatRequesty

MODEL_NAME = "openai/gpt-4o-mini"

requires_api_key = pytest.mark.skipif(
    not os.environ.get("REQUESTY_API_KEY"),
    reason="REQUESTY_API_KEY not set",
)


@requires_api_key
class TestChatRequesty(ChatModelIntegrationTests):
    """Test `ChatRequesty` chat model against the live Requesty router."""

    @property
    def chat_model_class(self) -> type[ChatRequesty]:
        """Return class of chat model being tested."""
        return ChatRequesty

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "temperature": 0,
        }

    @property
    def supports_json_mode(self) -> bool:
        """(bool) whether the chat model supports JSON mode."""
        return True
