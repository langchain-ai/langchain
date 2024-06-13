"""Test Redis cache functionality."""

import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, List, Optional, cast

import pytest
from langchain.globals import get_llm_cache, set_llm_cache
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from langchain_community.cache import AsyncRedisCache, RedisCache, RedisSemanticCache
from tests.integration_tests.cache.fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)
from tests.unit_tests.llms.fake_chat_model import FakeChatModel
from tests.unit_tests.llms.fake_llm import FakeLLM

# Using a non-standard port to avoid conflicts with potentially local running
# redis instances
# You can spin up a local redis using docker compose
# cd [repository-root]/docker
# docker-compose up redis
REDIS_TEST_URL = "redis://localhost:6020"


def random_string() -> str:
    return str(uuid.uuid4())


@contextmanager
def get_sync_redis(*, ttl: Optional[int] = 1) -> Generator[RedisCache, None, None]:
    """Get a sync RedisCache instance."""
    import redis

    cache = RedisCache(redis_=redis.Redis.from_url(REDIS_TEST_URL), ttl=ttl)
    try:
        yield cache
    finally:
        cache.clear()


@asynccontextmanager
async def get_async_redis(
    *, ttl: Optional[int] = 1
) -> AsyncGenerator[AsyncRedisCache, None]:
    """Get an async RedisCache instance."""
    from redis.asyncio import Redis

    cache = AsyncRedisCache(redis_=Redis.from_url(REDIS_TEST_URL), ttl=ttl)
    try:
        yield cache
    finally:
        await cache.aclear()


def test_redis_cache_ttl() -> None:
    from redis import Redis

    with get_sync_redis() as llm_cache:
        set_llm_cache(llm_cache)
        llm_cache.update("foo", "bar", [Generation(text="fizz")])
        key = llm_cache._key("foo", "bar")
        assert isinstance(llm_cache.redis, Redis)
        assert llm_cache.redis.pttl(key) > 0


async def test_async_redis_cache_ttl() -> None:
    from redis.asyncio import Redis as AsyncRedis

    async with get_async_redis() as redis_cache:
        set_llm_cache(redis_cache)
        llm_cache = cast(RedisCache, get_llm_cache())
        await llm_cache.aupdate("foo", "bar", [Generation(text="fizz")])
        key = llm_cache._key("foo", "bar")
        assert isinstance(llm_cache.redis, AsyncRedis)
        assert await llm_cache.redis.pttl(key) > 0


def test_sync_redis_cache() -> None:
    with get_sync_redis() as llm_cache:
        set_llm_cache(llm_cache)
        llm = FakeLLM()
        params = llm.dict()
        params["stop"] = None
        llm_string = str(sorted([(k, v) for k, v in params.items()]))
        llm_cache.update("prompt", llm_string, [Generation(text="fizz0")])
        output = llm.generate(["prompt"])
        expected_output = LLMResult(
            generations=[[Generation(text="fizz0")]],
            llm_output={},
        )
        assert output == expected_output


async def test_sync_in_async_redis_cache() -> None:
    """Test the sync RedisCache invoked with async methods"""
    with get_sync_redis() as llm_cache:
        set_llm_cache(llm_cache)
        llm = FakeLLM()
        params = llm.dict()
        params["stop"] = None
        llm_string = str(sorted([(k, v) for k, v in params.items()]))
        # llm_cache.update("meow", llm_string, [Generation(text="meow")])
        await llm_cache.aupdate("prompt", llm_string, [Generation(text="fizz1")])
        output = await llm.agenerate(["prompt"])
        expected_output = LLMResult(
            generations=[[Generation(text="fizz1")]],
            llm_output={},
        )
        assert output == expected_output


async def test_async_redis_cache() -> None:
    async with get_async_redis() as redis_cache:
        set_llm_cache(redis_cache)
        llm = FakeLLM()
        params = llm.dict()
        params["stop"] = None
        llm_string = str(sorted([(k, v) for k, v in params.items()]))
        llm_cache = cast(RedisCache, get_llm_cache())
        await llm_cache.aupdate("prompt", llm_string, [Generation(text="fizz2")])
        output = await llm.agenerate(["prompt"])
        expected_output = LLMResult(
            generations=[[Generation(text="fizz2")]],
            llm_output={},
        )
        assert output == expected_output


async def test_async_in_sync_redis_cache() -> None:
    async with get_async_redis() as redis_cache:
        set_llm_cache(redis_cache)
        llm = FakeLLM()
        params = llm.dict()
        params["stop"] = None
        llm_string = str(sorted([(k, v) for k, v in params.items()]))
        llm_cache = cast(RedisCache, get_llm_cache())
        with pytest.raises(NotImplementedError):
            llm_cache.update("foo", llm_string, [Generation(text="fizz")])


def test_redis_cache_chat() -> None:
    with get_sync_redis() as redis_cache:
        set_llm_cache(redis_cache)
        llm = FakeChatModel()
        params = llm.dict()
        params["stop"] = None
        llm_string = str(sorted([(k, v) for k, v in params.items()]))
        prompt: List[BaseMessage] = [HumanMessage(content="foo")]
        llm_cache = cast(RedisCache, get_llm_cache())
        llm_cache.update(
            dumps(prompt),
            llm_string,
            [ChatGeneration(message=AIMessage(content="fizz"))],
        )
        output = llm.generate([prompt])
        expected_output = LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="fizz"))]],
            llm_output={},
        )
        assert output == expected_output


async def test_async_redis_cache_chat() -> None:
    async with get_async_redis() as redis_cache:
        set_llm_cache(redis_cache)
        llm = FakeChatModel()
        params = llm.dict()
        params["stop"] = None
        llm_string = str(sorted([(k, v) for k, v in params.items()]))
        prompt: List[BaseMessage] = [HumanMessage(content="foo")]
        llm_cache = cast(RedisCache, get_llm_cache())
        await llm_cache.aupdate(
            dumps(prompt),
            llm_string,
            [ChatGeneration(message=AIMessage(content="fizz"))],
        )
        output = await llm.agenerate([prompt])
        expected_output = LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="fizz"))]],
            llm_output={},
        )
        assert output == expected_output


def test_redis_semantic_cache() -> None:
    """Test redis semantic cache functionality."""
    set_llm_cache(
        RedisSemanticCache(
            embedding=FakeEmbeddings(), redis_url=REDIS_TEST_URL, score_threshold=0.1
        )
    )
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    llm_cache = cast(RedisSemanticCache, get_llm_cache())
    llm_cache.update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(
        ["bar"]
    )  # foo and bar will have the same embedding produced by FakeEmbeddings
    expected_output = LLMResult(
        generations=[[Generation(text="fizz")]],
        llm_output={},
    )
    assert output == expected_output
    # clear the cache
    llm_cache.clear(llm_string=llm_string)
    output = llm.generate(
        ["bar"]
    )  # foo and bar will have the same embedding produced by FakeEmbeddings
    # expect different output now without cached result
    assert output != expected_output
    llm_cache.clear(llm_string=llm_string)


def test_redis_semantic_cache_multi() -> None:
    set_llm_cache(
        RedisSemanticCache(
            embedding=FakeEmbeddings(), redis_url=REDIS_TEST_URL, score_threshold=0.1
        )
    )
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    llm_cache = cast(RedisSemanticCache, get_llm_cache())
    llm_cache.update(
        "foo", llm_string, [Generation(text="fizz"), Generation(text="Buzz")]
    )
    output = llm.generate(
        ["bar"]
    )  # foo and bar will have the same embedding produced by FakeEmbeddings
    expected_output = LLMResult(
        generations=[[Generation(text="fizz"), Generation(text="Buzz")]],
        llm_output={},
    )
    assert output == expected_output
    # clear the cache
    llm_cache.clear(llm_string=llm_string)


def test_redis_semantic_cache_chat() -> None:
    set_llm_cache(
        RedisSemanticCache(
            embedding=FakeEmbeddings(), redis_url=REDIS_TEST_URL, score_threshold=0.1
        )
    )
    llm = FakeChatModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    prompt: List[BaseMessage] = [HumanMessage(content="foo")]
    llm_cache = cast(RedisSemanticCache, get_llm_cache())
    llm_cache.update(
        dumps(prompt), llm_string, [ChatGeneration(message=AIMessage(content="fizz"))]
    )
    output = llm.generate([prompt])
    expected_output = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="fizz"))]],
        llm_output={},
    )
    assert output == expected_output
    llm_cache.clear(llm_string=llm_string)


@pytest.mark.parametrize("embedding", [ConsistentFakeEmbeddings()])
@pytest.mark.parametrize(
    "prompts,  generations",
    [
        # Single prompt, single generation
        ([random_string()], [[random_string()]]),
        # Single prompt, multiple generations
        ([random_string()], [[random_string(), random_string()]]),
        # Single prompt, multiple generations
        ([random_string()], [[random_string(), random_string(), random_string()]]),
        # Multiple prompts, multiple generations
        (
            [random_string(), random_string()],
            [[random_string()], [random_string(), random_string()]],
        ),
    ],
    ids=[
        "single_prompt_single_generation",
        "single_prompt_multiple_generations",
        "single_prompt_multiple_generations",
        "multiple_prompts_multiple_generations",
    ],
)
def test_redis_semantic_cache_hit(
    embedding: Embeddings, prompts: List[str], generations: List[List[str]]
) -> None:
    set_llm_cache(RedisSemanticCache(embedding=embedding, redis_url=REDIS_TEST_URL))

    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))

    llm_generations = [
        [
            Generation(text=generation, generation_info=params)
            for generation in prompt_i_generations
        ]
        for prompt_i_generations in generations
    ]
    llm_cache = cast(RedisSemanticCache, get_llm_cache())
    for prompt_i, llm_generations_i in zip(prompts, llm_generations):
        print(prompt_i)  # noqa: T201
        print(llm_generations_i)  # noqa: T201
        llm_cache.update(prompt_i, llm_string, llm_generations_i)
    llm.generate(prompts)
    assert llm.generate(prompts) == LLMResult(
        generations=llm_generations, llm_output={}
    )
