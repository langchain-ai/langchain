"""Test Upstash Redis cache functionality."""
import uuid

import pytest

import langchain
from langchain.cache import UpstashRedisCache
from langchain.schema import Generation, LLMResult
from tests.unit_tests.llms.fake_chat_model import FakeChatModel
from tests.unit_tests.llms.fake_llm import FakeLLM

URL = "<UPSTASH_REDIS_REST_URL>"
TOKEN = "<UPSTASH_REDIS_REST_TOKEN>"


def random_string() -> str:
    return str(uuid.uuid4())


@pytest.mark.requires("upstash_redis")
def test_redis_cache_ttl() -> None:
    from upstash_redis import Redis

    langchain.llm_cache = UpstashRedisCache(redis_=Redis(url=URL, token=TOKEN), ttl=1)
    langchain.llm_cache.update("foo", "bar", [Generation(text="fizz")])
    key = langchain.llm_cache._key("foo", "bar")
    assert langchain.llm_cache.redis.pttl(key) > 0


@pytest.mark.requires("upstash_redis")
def test_redis_cache() -> None:
    from upstash_redis import Redis

    langchain.llm_cache = UpstashRedisCache(redis_=Redis(url=URL, token=TOKEN), ttl=1)
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

    lookup_output = langchain.llm_cache.lookup("foo", llm_string)
    if lookup_output and len(lookup_output) > 0:
        assert lookup_output == expected_output.generations[0]

    langchain.llm_cache.clear()
    output = llm.generate(["foo"])

    assert output != expected_output
    langchain.llm_cache.redis.flushall()


def test_redis_cache_multi() -> None:
    from upstash_redis import Redis

    langchain.llm_cache = UpstashRedisCache(redis_=Redis(url=URL, token=TOKEN), ttl=1)
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    langchain.llm_cache.update(
        "foo", llm_string, [Generation(text="fizz"), Generation(text="Buzz")]
    )
    output = llm.generate(
        ["foo"]
    )  # foo and bar will have the same embedding produced by FakeEmbeddings
    expected_output = LLMResult(
        generations=[[Generation(text="fizz"), Generation(text="Buzz")]],
        llm_output={},
    )
    assert output == expected_output
    # clear the cache
    langchain.llm_cache.clear()


@pytest.mark.requires("upstash_redis")
def test_redis_cache_chat() -> None:
    from upstash_redis import Redis

    langchain.llm_cache = UpstashRedisCache(redis_=Redis(url=URL, token=TOKEN), ttl=1)
    llm = FakeChatModel()
    params = llm.dict()
    params["stop"] = None
    with pytest.warns():
        llm.predict("foo")
    langchain.llm_cache.redis.flushall()
