import asyncio
from itertools import cycle
from typing import AsyncGenerator, Generator, List

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import FakeListLLM, GenericFakeChatModel
from langchain_core.load.dump import dumps
from langchain_core.messages import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_openai.embeddings.base import OpenAIEmbeddings
from ulid import ULID

from langchain_redis import RedisCache, RedisSemanticCache


def random_string() -> str:
    return str(ULID())


class DummyEmbeddings(Embeddings):
    def __init__(self, dims: int = 3):
        self.dims = dims

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * self.dims for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * self.dims


@pytest.fixture
def redis_url() -> str:
    return "redis://localhost:6379"


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(size=768)


@pytest.fixture
def openai_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings()


@pytest.fixture
def redis_cache(redis_url: str) -> Generator[RedisCache, None, None]:
    cache = RedisCache(redis_url=redis_url, ttl=3600)  # Set TTL to 1 hour
    set_llm_cache(cache)
    try:
        yield cache
    finally:
        cache.clear()


@pytest.fixture
async def async_redis_cache(redis_url: str) -> AsyncGenerator[RedisCache, None]:
    cache = RedisCache(redis_url=redis_url, ttl=3600)  # Set TTL to 1 hour
    set_llm_cache(cache)
    try:
        yield cache
    finally:
        await cache.aclear()


@pytest.fixture(scope="function")
def redis_semantic_cache(
    openai_embeddings: OpenAIEmbeddings, redis_url: str
) -> Generator[RedisSemanticCache, None, None]:
    cache = RedisSemanticCache(
        name=f"semcache_{str(ULID())}",
        redis_url=redis_url,
        embeddings=openai_embeddings,
    )
    try:
        yield cache
    finally:
        cache.clear()


class TestRedisCacheBasicIntegration:
    def test_redis_cache(self, redis_cache: RedisCache) -> None:
        redis_cache.update(
            "test_prompt", "test_llm", [Generation(text="test_response")]
        )
        result = redis_cache.lookup("test_prompt", "test_llm")
        assert result is not None
        assert len(result) == 1
        assert result[0].text == "test_response"
        redis_cache.clear()
        assert redis_cache.lookup("test_prompt", "test_llm") is None

    def test_redis_cache_ttl(self, redis_cache: RedisCache) -> None:
        llm = FakeListLLM(cache=redis_cache, responses=["foo", "bar"])
        prompt = random_string()
        redis_cache.update(prompt, str(llm), [Generation(text="test response")])

        # Check that the TTL is set
        key = redis_cache._key(prompt, str(llm))
        ttl_result = redis_cache.redis.ttl(key)
        if asyncio.iscoroutine(ttl_result):
            ttl = asyncio.get_event_loop().run_until_complete(ttl_result)
        else:
            ttl = ttl_result
        assert 0 < ttl <= 3600

    @pytest.mark.asyncio
    async def test_async_redis_cache(self, async_redis_cache: RedisCache) -> None:
        llm = FakeListLLM(cache=async_redis_cache, responses=["foo", "bar"])
        prompt = random_string()
        await async_redis_cache.aupdate(
            prompt, str(llm), [Generation(text="async test")]
        )

        result = await async_redis_cache.alookup(prompt, str(llm))
        assert result == [Generation(text="async test")]

    @pytest.mark.asyncio
    async def test_async_redis_cache_clear(self, async_redis_cache: RedisCache) -> None:
        llm = FakeListLLM(cache=async_redis_cache, responses=["foo", "bar"])
        prompt = random_string()
        await async_redis_cache.aupdate(
            prompt, str(llm), [Generation(text="async test")]
        )

        await async_redis_cache.aclear()

        result = await async_redis_cache.alookup(prompt, str(llm))
        assert result is None

    def test_redis_cache_chat(self, redis_cache: RedisCache) -> None:
        responses = cycle(
            [
                AIMessage(content="Hello from cache"),
                AIMessage(content="How are you from cache"),
            ]
        )
        chat_model = GenericFakeChatModel(messages=responses)

        human_message1 = HumanMessage(content="Hello")
        human_message2 = HumanMessage(content="How are you?")
        prompt1: List[BaseMessage] = [human_message1]
        prompt2: List[BaseMessage] = [human_message2]

        # First call should generate a response and cache it
        result1 = chat_model.generate([prompt1])

        # Instead of comparing the entire LLMResult, let's check specific parts
        assert len(result1.generations) == 1
        assert len(result1.generations[0]) == 1
        assert isinstance(result1.generations[0][0], ChatGeneration)
        assert result1.generations[0][0].message.content == "Hello from cache"

        # Cache the result manually (since GenericFakeChatModel doesn't use the cache
        # internally)
        redis_cache.update(dumps(prompt1), str(chat_model), result1.generations[0])

        # Second call with the same prompt should hit the cache
        cached_result = redis_cache.lookup(dumps(prompt1), str(chat_model))
        assert cached_result is not None
        assert isinstance(cached_result[0], ChatGeneration)
        assert cached_result[0].message.content == "Hello from cache"

        # Call with a different prompt should generate a new response
        result3 = chat_model.generate([prompt2])

        # Check specific parts of the new result
        assert len(result3.generations) == 1
        assert len(result3.generations[0]) == 1
        assert isinstance(result3.generations[0][0], ChatGeneration)
        assert result3.generations[0][0].message.content == "How are you from cache"

        # Cache the new result
        redis_cache.update(dumps(prompt2), str(chat_model), result3.generations[0])

        # Verify that both prompts are in the cache
        cached_result1 = redis_cache.lookup(dumps(prompt1), str(chat_model))
        cached_result2 = redis_cache.lookup(dumps(prompt2), str(chat_model))
        assert cached_result1 is not None and cached_result2 is not None
        assert isinstance(cached_result1[0], ChatGeneration)
        assert isinstance(cached_result2[0], ChatGeneration)
        assert cached_result1[0].message.content == "Hello from cache"
        assert cached_result2[0].message.content == "How are you from cache"
