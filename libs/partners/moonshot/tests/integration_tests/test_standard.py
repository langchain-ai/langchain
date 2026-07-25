"""Standard integration tests for Moonshot chat model."""

from __future__ import annotations

import os

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_moonshot.chat_models import ChatMoonshot

moonshot_api_base = os.environ.get("MOONSHOT_API_BASE", "")

if "deepseek" in moonshot_api_base.lower():
    _MODEL = "deepseek-v4-flash"
else:
    _MODEL = "moonshot-v1-8k"


class TestMoonshotIntegration(ChatModelIntegrationTests):
    """Standard ChatModel integration tests for Moonshot."""

    @property
    def chat_model_class(self) -> type[ChatMoonshot]:
        return ChatMoonshot

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": _MODEL,
        }
