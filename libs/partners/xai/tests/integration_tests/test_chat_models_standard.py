"""Standard LangChain interface tests"""

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_tests.integration_tests import (  # type: ignore[import-not-found]
    ChatModelIntegrationTests,  # type: ignore[import-not-found]
)

from langchain_xai import ChatXAI

# Initialize the rate limiter in global scope, so it can be re-used
# across tests.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
)


class TestXAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatXAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "grok-3",
            "rate_limiter": rate_limiter,
            "stream_usage": True,
        }


def test_reasoning_content() -> None:
    """Test reasoning content."""
    chat_model = ChatXAI(
        model="grok-3-mini-beta",
        reasoning_effort="low",
    )
    response = chat_model.invoke("What is 3^3?")
    assert response.content
    assert response.additional_kwargs["reasoning_content"]

    # Test streaming
    full: Optional[BaseMessageChunk] = None
    for chunk in chat_model.stream("What is 3^3?"):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs["reasoning_content"]
