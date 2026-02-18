"""Tests for embedding model rate limiting."""

import asyncio
import threading
import time

import pytest
from blockbuster import BlockBuster

from langchain_core.embeddings import DeterministicFakeEmbedding, FakeEmbeddings
from langchain_core.rate_limiters import InMemoryRateLimiter


@pytest.fixture(autouse=True)
def deactivate_blockbuster(blockbuster: BlockBuster) -> None:
    # Deactivate BlockBuster to not disturb the rate limiter timings
    blockbuster.deactivate()


def test_rate_limit_embed_documents() -> None:
    """Test that embed_documents respects rate limiter timing."""
    model = FakeEmbeddings(
        size=10,
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=20,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        ),
    )
    tic = time.time()
    result = model.embed_documents(["hello", "world"])
    toc = time.time()
    # Should wait for rate limiter token (bucket starts empty)
    assert 0.10 < toc - tic < 0.20
    assert len(result) == 2
    assert len(result[0]) == 10


def test_rate_limit_embed_query() -> None:
    """Test that embed_query respects rate limiter timing."""
    model = FakeEmbeddings(
        size=10,
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=20,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        ),
    )
    tic = time.time()
    result = model.embed_query("hello")
    toc = time.time()
    assert 0.10 < toc - tic < 0.20
    assert len(result) == 10


async def test_rate_limit_aembed_documents() -> None:
    """Test that aembed_documents respects rate limiter timing."""
    model = FakeEmbeddings(
        size=10,
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=20,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        ),
    )
    tic = time.time()
    result = await model.aembed_documents(["hello", "world"])
    toc = time.time()
    assert 0.10 < toc - tic < 0.20
    assert len(result) == 2


async def test_rate_limit_aembed_query() -> None:
    """Test that aembed_query respects rate limiter timing."""
    model = FakeEmbeddings(
        size=10,
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=20,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        ),
    )
    tic = time.time()
    result = await model.aembed_query("hello")
    toc = time.time()
    assert 0.10 < toc - tic < 0.20
    assert len(result) == 10


def test_no_rate_limit_by_default() -> None:
    """Test that embeddings work normally without rate limiter."""
    model = FakeEmbeddings(size=10)
    tic = time.time()
    result = model.embed_documents(["hello", "world"])
    toc = time.time()
    # Should be fast without rate limiting
    assert toc - tic < 0.05
    assert len(result) == 2


def test_with_rate_limiter_wrapper() -> None:
    """Test with_rate_limiter() creates a working rate-limited wrapper."""
    base = DeterministicFakeEmbedding(size=10)
    limiter = InMemoryRateLimiter(
        requests_per_second=20,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    )
    wrapped = base.with_rate_limiter(limiter)

    # First call should wait for token
    tic = time.time()
    result = wrapped.embed_documents(["hello"])
    toc = time.time()
    assert 0.10 < toc - tic < 0.20
    assert len(result) == 1
    assert len(result[0]) == 10


async def test_with_rate_limiter_wrapper_async() -> None:
    """Test with_rate_limiter() works for async methods."""
    base = DeterministicFakeEmbedding(size=10)
    limiter = InMemoryRateLimiter(
        requests_per_second=20,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    )
    wrapped = base.with_rate_limiter(limiter)

    tic = time.time()
    result = await wrapped.aembed_query("hello")
    toc = time.time()
    assert 0.10 < toc - tic < 0.20
    assert len(result) == 10


def test_with_rate_limiter_preserves_results() -> None:
    """Test that the wrapper produces the same results as the base embeddings."""
    base = DeterministicFakeEmbedding(size=10)
    limiter = InMemoryRateLimiter(
        requests_per_second=100,
        check_every_n_seconds=0.01,
        max_bucket_size=100,
    )
    wrapped = base.with_rate_limiter(limiter)

    # Allow rate limiter to warm up
    time.sleep(0.05)

    base_result = base.embed_query("test text")
    wrapped_result = wrapped.embed_query("test text")
    assert base_result == wrapped_result

    base_docs = base.embed_documents(["doc1", "doc2"])
    wrapped_docs = wrapped.embed_documents(["doc1", "doc2"])
    assert base_docs == wrapped_docs


def test_concurrent_rate_limiting() -> None:
    """Test that rate limiting works under concurrent access."""
    model = FakeEmbeddings(
        size=10,
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=10,
            check_every_n_seconds=0.01,
            max_bucket_size=1,
        ),
    )

    results: list[list[list[float]]] = []
    errors: list[Exception] = []

    def embed_worker() -> None:
        try:
            result = model.embed_documents(["text"])
            results.append(result)
        except Exception as e:
            errors.append(e)

    tic = time.time()
    threads = [threading.Thread(target=embed_worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    toc = time.time()

    assert len(errors) == 0
    assert len(results) == 3
    # 3 calls at 10 req/s with bucket size 1 → ~0.2s minimum
    assert toc - tic >= 0.15


async def test_concurrent_async_rate_limiting() -> None:
    """Test that async rate limiting works under concurrent access."""
    model = FakeEmbeddings(
        size=10,
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=10,
            check_every_n_seconds=0.01,
            max_bucket_size=1,
        ),
    )

    tic = time.time()
    results = await asyncio.gather(
        model.aembed_documents(["text1"]),
        model.aembed_documents(["text2"]),
        model.aembed_documents(["text3"]),
    )
    toc = time.time()

    assert len(results) == 3
    for result in results:
        assert len(result) == 1
    # 3 concurrent calls at 10 req/s with bucket size 1
    assert toc - tic >= 0.15


def test_deterministic_fake_with_rate_limiter() -> None:
    """Test DeterministicFakeEmbedding with rate limiter field."""
    model = DeterministicFakeEmbedding(
        size=10,
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=20,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        ),
    )
    tic = time.time()
    result = model.embed_documents(["hello"])
    toc = time.time()
    assert 0.10 < toc - tic < 0.20
    assert len(result) == 1


def test_second_call_faster() -> None:
    """Test that second call is faster due to token accumulation."""
    model = FakeEmbeddings(
        size=10,
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=20,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        ),
    )

    # First call waits for token
    tic = time.time()
    model.embed_documents(["hello"])
    toc = time.time()
    assert 0.10 < toc - tic < 0.20

    # Second call should be faster (token accumulated during first call)
    tic = time.time()
    model.embed_documents(["world"])
    toc = time.time()
    assert toc - tic < 0.10
