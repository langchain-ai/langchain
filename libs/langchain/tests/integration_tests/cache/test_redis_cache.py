"""Test Redis cache functionality."""
import uuid
from typing import List

import pytest

import langchain
from langchain.cache import RedisCache, RedisSemanticCache
from langchain.embeddings.base import Embeddings
from langchain.schema import Generation, LLMResult
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)
from tests.unit_tests.llms.fake_chat_model import FakeChatModel
from tests.unit_tests.llms.fake_llm import FakeLLM

REDIS_TEST_URL = "redis://localhost:6379"


def random_string() -> str:
    return str(uuid.uuid4())


def test_redis_cache_ttl() -> None:
    import redis

    langchain.llm_cache = RedisCache(redis_=redis.Redis.from_url(REDIS_TEST_URL), ttl=1)
    langchain.llm_cache.update("foo", "bar", [Generation(text="fizz")])
    key = langchain.llm_cache._key("foo", "bar")
    assert langchain.llm_cache.redis.pttl(key) > 0


def test_redis_cache() -> None:
    import redis

    langchain.llm_cache = RedisCache(redis_=redis.Redis.from_url(REDIS_TEST_URL))
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    langchain.llm_cache.update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(["foo"])
    expected_output = LLMResult(
        generations=[[Generation(text="fizz")]],
        llm_output={},
    )
    assert output == expected_output
    langchain.llm_cache.redis.flushall()


def test_redis_cache_chat() -> None:
    import redis

    langchain.llm_cache = RedisCache(redis_=redis.Redis.from_url(REDIS_TEST_URL))
    llm = FakeChatModel()
    params = llm.dict()
    params["stop"] = None
    with pytest.warns():
        llm.predict("foo")
    llm.predict("foo")
    langchain.llm_cache.redis.flushall()


def test_redis_semantic_cache() -> None:
    langchain.llm_cache = RedisSemanticCache(
        embedding=FakeEmbeddings(), redis_url=REDIS_TEST_URL, score_threshold=0.1
    )
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    langchain.llm_cache.update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(
        ["bar"]
    )  # foo and bar will have the same embedding produced by FakeEmbeddings
    expected_output = LLMResult(
        generations=[[Generation(text="fizz")]],
        llm_output={},
    )
    assert output == expected_output
    # clear the cache
    langchain.llm_cache.clear(llm_string=llm_string)
    output = llm.generate(
        ["bar"]
    )  # foo and bar will have the same embedding produced by FakeEmbeddings
    # expect different output now without cached result
    assert output != expected_output
    langchain.llm_cache.clear(llm_string=llm_string)


def test_redis_semantic_cache_multi() -> None:
    langchain.llm_cache = RedisSemanticCache(
        embedding=FakeEmbeddings(), redis_url=REDIS_TEST_URL, score_threshold=0.1
    )
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    langchain.llm_cache.update(
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
    langchain.llm_cache.clear(llm_string=llm_string)


def test_redis_semantic_cache_chat() -> None:
    langchain.llm_cache = RedisSemanticCache(
        embedding=FakeEmbeddings(), redis_url=REDIS_TEST_URL, score_threshold=0.1
    )
    llm = FakeChatModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    with pytest.warns():
        llm.predict("foo")
    llm.predict("foo")
    langchain.llm_cache.clear(llm_string=llm_string)


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
    langchain.llm_cache = RedisSemanticCache(
        embedding=embedding, redis_url=REDIS_TEST_URL
    )

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
    for prompt_i, llm_generations_i in zip(prompts, llm_generations):
        print(prompt_i)
        print(llm_generations_i)
        langchain.llm_cache.update(prompt_i, llm_string, llm_generations_i)
    llm.generate(prompts)
    assert llm.generate(prompts) == LLMResult(
        generations=llm_generations, llm_output={}
    )
