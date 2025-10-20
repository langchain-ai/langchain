"""Standard LangChain interface tests"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_tests.integration_tests import (  # type: ignore[import-not-found]
    ChatModelIntegrationTests,  # type: ignore[import-not-found]
)
from typing_extensions import override

from langchain_xai import ChatXAI

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

# Initialize the rate limiter in global scope, so it can be re-used across tests
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
)

MODEL_NAME = "grok-4-fast-reasoning"


class TestXAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatXAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": MODEL_NAME,
            "rate_limiter": rate_limiter,
            "stream_usage": True,
        }

    @pytest.mark.xfail(
        reason="Default model does not support stop sequences, using grok-3 instead"
    )
    @override
    def test_stop_sequence(self, model: BaseChatModel) -> None:
        """Override to use `grok-3` which supports stop sequences."""
        params = {**self.chat_model_params, "model": "grok-3"}

        grok3_model = ChatXAI(**params)

        result = grok3_model.invoke("hi", stop=["you"])
        assert isinstance(result, AIMessage)

        custom_model = ChatXAI(
            **params,
            stop_sequences=["you"],
        )
        result = custom_model.invoke("hi")
        assert isinstance(result, AIMessage)
