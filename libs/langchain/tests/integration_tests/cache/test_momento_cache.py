"""Test Momento cache functionality.

To run tests, set the environment variable MOMENTO_AUTH_TOKEN to a valid
Momento auth token. This can be obtained by signing up for a free
Momento account at https://gomomento.com/.
"""
from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Iterator

import pytest

from langchain.cache import MomentoCache
from langchain.globals import set_llm_cache
from langchain.schema import Generation, LLMResult
from tests.unit_tests.llms.fake_llm import FakeLLM


def random_string() -> str:
    return str(uuid.uuid4())


@pytest.fixture(scope="module")
def momento_cache() -> Iterator[MomentoCache]:
    from momento import CacheClient, Configurations, CredentialProvider

    cache_name = f"langchain-test-cache-{random_string()}"
    client = CacheClient(
        Configurations.Laptop.v1(),
        CredentialProvider.from_environment_variable("MOMENTO_API_KEY"),
        default_ttl=timedelta(seconds=30),
    )
    try:
        llm_cache = MomentoCache(client, cache_name)
        set_llm_cache(llm_cache)
        yield llm_cache
    finally:
        client.delete_cache(cache_name)


def test_invalid_ttl() -> None:
    from momento import CacheClient, Configurations, CredentialProvider

    client = CacheClient(
        Configurations.Laptop.v1(),
        CredentialProvider.from_environment_variable("MOMENTO_API_KEY"),
        default_ttl=timedelta(seconds=30),
    )
    with pytest.raises(ValueError):
        MomentoCache(client, cache_name=random_string(), ttl=timedelta(seconds=-1))


def test_momento_cache_miss(momento_cache: MomentoCache) -> None:
    llm = FakeLLM()
    stub_llm_output = LLMResult(generations=[[Generation(text="foo")]])
    assert llm.generate([random_string()]) == stub_llm_output


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
)
def test_momento_cache_hit(
    momento_cache: MomentoCache, prompts: list[str], generations: list[list[str]]
) -> None:
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
        momento_cache.update(prompt_i, llm_string, llm_generations_i)

    assert llm.generate(prompts) == LLMResult(
        generations=llm_generations, llm_output={}
    )
