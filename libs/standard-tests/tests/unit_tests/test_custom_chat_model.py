"""Test the standard tests on the custom chat model in the docs."""

from __future__ import annotations

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from langchain_tests.integration_tests import ChatModelIntegrationTests
from langchain_tests.unit_tests import ChatModelUnitTests

from .custom_chat_model import ChatParrotLink


class TestChatParrotLinkUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatParrotLink]:
        return ChatParrotLink

    @property
    def chat_model_params(self) -> dict:
        return {"model": "bird-brain-001", "temperature": 0, "parrot_buffer_length": 50}


class TestChatParrotLinkIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[ChatParrotLink]:
        return ChatParrotLink

    @property
    def chat_model_params(self) -> dict:
        return {"model": "bird-brain-001", "temperature": 0, "parrot_buffer_length": 50}

    @pytest.mark.xfail(reason="ChatParrotLink doesn't implement bind_tools method")
    def test_unicode_tool_call_integration(
        self,
        model: BaseChatModel,
        tool_choice: str | None = None,  # noqa: PT028
        force_tool_call: bool = True,  # noqa: FBT001, FBT002, PT028
    ) -> None:
        """Expected failure as ChatParrotLink doesn't support tool calling yet."""
