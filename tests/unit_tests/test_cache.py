"""Test caching for LLMs and ChatModels."""
import os
from typing import Any, Dict, Generator, Union

import pytest
from _pytest.fixtures import FixtureRequest

import langchain
from langchain.cache import (
    BaseCache,
    GPTCache,
    InMemoryCache,
    MomentoCache,
    RedisCache,
    SQLAlchemyCache,
    SQLiteCache,
)
from langchain.chat_models import FakeListChatModel
from langchain.chat_models.base import BaseChatModel, chat_history_as_string
from langchain.llms import FakeListLLM
from langchain.llms.base import BaseLLM
from langchain.schema import AIMessage, ChatGeneration, Generation, _message_from_dict

CACHE_DB_FILE = "cache.test.db"

CACHE_OPTIONS = [
    ("memory", InMemoryCache()),
    ("sqlite", SQLiteCache(CACHE_DB_FILE)),
    # - Redis Cache
    # - GPTCache
    # - Momento Cache
    # - SQLAlchemy Cache
]


@pytest.fixture(autouse=True, params=CACHE_OPTIONS)
def set_cache_and_teardown(request: FixtureRequest) -> Generator[None, None, None]:
    # Will be run before each test
    cache_type, cache_instance = request.param
    if cache_type == "sqlite":
        if os.path.exists(CACHE_DB_FILE):
            os.remove(CACHE_DB_FILE)
    langchain.llm_cache = cache_instance

    yield

    # Will be run after each test
    if cache_type == "sqlite":
        if os.path.exists(CACHE_DB_FILE):
            os.remove(CACHE_DB_FILE)


def test_llm_caching() -> None:
    prompt = "How are you?"
    response = "Test response"
    cached_response = "Cached test response"
    llm = FakeListLLM(responses=[response])
    if isinstance(langchain.llm_cache, BaseCache):
        langchain.llm_cache.update(
            prompt=prompt,
            llm_string=create_llm_string(llm),
            return_val=[Generation(text=cached_response)],
        )
        assert llm(prompt) == cached_response
    else:
        raise ValueError(
            "The cache not set. This should never happen, as the pytest fixture "
            "`set_cache` always sets the cache."
        )


def test_chat_model_caching() -> None:
    prompt = [
        _message_from_dict({"type": "human", "data": {"content": "How are you?"}})
    ]
    response = "Test response"
    cached_response = "Cached test response"
    cached_message = _message_from_dict(
        {"type": "ai", "data": {"content": cached_response}}
    )
    llm = FakeListChatModel(responses=[response])
    if isinstance(langchain.llm_cache, BaseCache):
        langchain.llm_cache.update(
            prompt=chat_history_as_string(prompt),
            llm_string=create_llm_string(llm),
            return_val=[ChatGeneration(text=cached_response, message=cached_message)],
        )
        result = llm(prompt)
        assert isinstance(result, AIMessage)
        assert llm(prompt).content == cached_response
    else:
        raise ValueError(
            "The cache not set. This should never happen, as the pytest fixture "
            "`set_cache` always sets the cache."
        )


def create_llm_string(llm: Union[BaseLLM, BaseChatModel]) -> str:
    _dict: Dict = llm.dict()
    _dict["stop"] = None
    return str(sorted([(k, v) for k, v in _dict.items()]))
