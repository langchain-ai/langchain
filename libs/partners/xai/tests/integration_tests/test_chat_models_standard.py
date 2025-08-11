"""Standard LangChain interface tests"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_tests.integration_tests import (  # type: ignore[import-not-found]
    ChatModelIntegrationTests,  # type: ignore[import-not-found]
)

from langchain_xai import ChatXAI

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

# Initialize the rate limiter in global scope, so it can be re-used
# across tests.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
)


# Not using Grok 4 since it doesn't support reasoning params (effort) or returns
# reasoning content.


class TestXAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatXAI

    @property
    def chat_model_params(self) -> dict:
        # TODO: bump to test new Grok once they implement other features
        return {
            "model": "grok-3",
            "rate_limiter": rate_limiter,
            "stream_usage": True,
        }


def test_reasoning_content() -> None:
    """Test reasoning content."""
    chat_model = ChatXAI(
        model="grok-3-mini",
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


def test_web_search() -> None:
    llm = ChatXAI(
        model="grok-3",
        search_parameters={"mode": "auto", "max_search_results": 3},
    )

    # Test invoke
    response = llm.invoke("Provide me a digest of world news in the last 24 hours.")
    assert response.content
    assert response.additional_kwargs["citations"]
    assert len(response.additional_kwargs["citations"]) <= 3

    # Test streaming
    full = None
    for chunk in llm.stream("Provide me a digest of world news in the last 24 hours."):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs["citations"]
    assert len(full.additional_kwargs["citations"]) <= 3
