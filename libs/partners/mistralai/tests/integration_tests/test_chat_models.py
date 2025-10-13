"""Test ChatMistral chat model."""

from __future__ import annotations

import logging
import time

import pytest
from httpx import ReadTimeout
from langchain_core.messages import AIMessageChunk, BaseMessageChunk

from langchain_mistralai.chat_models import ChatMistralAI


async def test_astream() -> None:
    """Test streaming tokens from ChatMistralAI."""
    llm = ChatMistralAI()

    full: BaseMessageChunk | None = None
    chunks_with_token_counts = 0
    chunks_with_response_metadata = 0
    async for token in llm.astream("Hello"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
        full = token if full is None else full + token
        if token.usage_metadata is not None:
            chunks_with_token_counts += 1
        if token.response_metadata:
            chunks_with_response_metadata += 1
    if chunks_with_token_counts != 1 or chunks_with_response_metadata != 1:
        msg = (
            "Expected exactly one chunk with token counts or response_metadata. "
            "AIMessageChunk aggregation adds / appends counts and metadata. Check that "
            "this is behaving properly."
        )
        raise AssertionError(msg)
    assert isinstance(full, AIMessageChunk)
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert (
        full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
        == full.usage_metadata["total_tokens"]
    )
    assert isinstance(full.response_metadata["model_name"], str)
    assert full.response_metadata["model_name"]


def test_retry_parameters(caplog: pytest.LogCaptureFixture) -> None:
    """Test that retry parameters are honored in ChatMistralAI."""
    # Create a model with intentionally short timeout and multiple retries
    mistral = ChatMistralAI(
        timeout=1,  # Very short timeout to trigger timeouts
        max_retries=3,  # Should retry 3 times
    )

    # Simple test input that should take longer than 1 second to process
    test_input = "Write a 2 sentence story about a cat"

    # Measure start time
    t0 = time.time()
    logger = logging.getLogger(__name__)

    try:
        # Try to get a response
        response = mistral.invoke(test_input)

        # If successful, validate the response
        elapsed_time = time.time() - t0
        logger.info("Request succeeded in %.2f seconds", elapsed_time)
        # Check that we got a valid response
        assert response.content
        assert isinstance(response.content, str)
        assert "cat" in response.content.lower()

    except ReadTimeout:
        elapsed_time = time.time() - t0
        logger.info("Request timed out after %.2f seconds", elapsed_time)
        assert elapsed_time >= 3.0
        pytest.skip("Test timed out as expected with short timeout")
    except Exception:
        logger.exception("Unexpected exception")
        raise
