"""Integration tests for OpenAI Batch API.

These tests require an ``OPENAI_API_KEY`` environment variable and will
make real API calls. The Batch API has a 24-hour completion window, so
these tests may take a while to complete (typically minutes, not hours,
for small batches).
"""

from __future__ import annotations

import os

import pytest

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models._batch_api import BatchResult

_SKIP_REASON = "Set OPENAI_API_KEY to run Batch API integration tests"


@pytest.fixture
def model() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", use_batch_api=True)


@pytest.fixture
def model_no_batch() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini")


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason=_SKIP_REASON)
class TestBatchApiEndToEnd:
    def test_batch_simple(self, model: ChatOpenAI) -> None:
        """Submit a small batch and verify results come back."""
        results = model.batch(
            [
                "Say 'hello' and nothing else.",
                "Say 'world' and nothing else.",
            ],
            config={"configurable": {"batch_poll_interval": 5}},
        )
        assert len(results) == 2
        for r in results:
            assert hasattr(r, "content")
            assert len(r.content) > 0

    def test_batch_with_return_exceptions(self, model: ChatOpenAI) -> None:
        """Verify return_exceptions works with batch API."""
        results = model.batch(
            ["Say 'test' and nothing else."],
            return_exceptions=True,
        )
        assert len(results) == 1
        # Should be a successful message, not an exception
        assert hasattr(results[0], "content")


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason=_SKIP_REASON)
class TestBatchApiLifecycle:
    def test_submit_status_retrieve(self, model_no_batch: ChatOpenAI) -> None:
        """Test the fire-and-forget lifecycle."""
        batch_id = model_no_batch.batch_api_submit(
            ["Say 'lifecycle test' and nothing else."]
        )
        assert isinstance(batch_id, str)
        assert len(batch_id) > 0

        status = model_no_batch.batch_api_status(batch_id)
        assert isinstance(status, BatchResult)
        assert status.batch_id == batch_id
        assert status.status in {
            "validating",
            "in_progress",
            "finalizing",
            "completed",
        }

    def test_cancel(self, model_no_batch: ChatOpenAI) -> None:
        """Test batch cancellation."""
        batch_id = model_no_batch.batch_api_submit(
            [f"Question {i}" for i in range(10)]
        )
        # Cancel immediately
        model_no_batch.batch_api_cancel(batch_id)

        status = model_no_batch.batch_api_status(batch_id)
        assert status.status in {"cancelling", "cancelled", "canceled"}


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason=_SKIP_REASON)
class TestBatchApiAsync:
    @pytest.mark.asyncio
    async def test_abatch_simple(self, model: ChatOpenAI) -> None:
        """Async batch submission and retrieval."""
        results = await model.abatch(
            ["Say 'async hello' and nothing else."],
            config={"configurable": {"batch_poll_interval": 5}},
        )
        assert len(results) == 1
        assert hasattr(results[0], "content")

    @pytest.mark.asyncio
    async def test_abatch_submit_and_retrieve(
        self, model_no_batch: ChatOpenAI
    ) -> None:
        """Async fire-and-forget lifecycle."""
        batch_id = await model_no_batch.abatch_api_submit(
            ["Say 'async lifecycle' and nothing else."]
        )
        assert isinstance(batch_id, str)
