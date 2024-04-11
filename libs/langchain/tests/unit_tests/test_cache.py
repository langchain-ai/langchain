"""Test caching for LLMs and ChatModels."""
import sqlite3
from typing import Dict, Generator, List, Union

import pytest
from _pytest.fixtures import FixtureRequest
from langchain_community.chat_models import FakeListChatModel
from langchain_community.llms import FakeListLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.load import dumps
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, Generation
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from langchain.cache import InMemoryCache, SQLAlchemyCache
from langchain.globals import get_llm_cache, set_llm_cache


def get_sqlite_cache() -> SQLAlchemyCache:
    return SQLAlchemyCache(
        engine=create_engine(
            "sqlite://", creator=lambda: sqlite3.connect("file::memory:?cache=shared")
        )
    )


CACHE_OPTIONS = [
    InMemoryCache,
    get_sqlite_cache,
]


@pytest.fixture(autouse=True, params=CACHE_OPTIONS)
def set_cache_and_teardown(request: FixtureRequest) -> Generator[None, None, None]:
    # Will be run before each test
    cache_instance = request.param
    set_llm_cache(cache_instance())
    if llm_cache := get_llm_cache():
        llm_cache.clear()
    else:
        raise ValueError("Cache not set. This should never happen.")

    yield

    # Will be run after each test
    if llm_cache:
        llm_cache.clear()
        set_llm_cache(None)
    else:
        raise ValueError("Cache not set. This should never happen.")


async def test_llm_caching() -> None:
    prompt = "How are you?"
    response = "Test response"
    cached_response = "Cached test response"
    llm = FakeListLLM(responses=[response])
    if llm_cache := get_llm_cache():
        # sync test
        llm_cache.update(
            prompt=prompt,
            llm_string=create_llm_string(llm),
            return_val=[Generation(text=cached_response)],
        )
        assert llm.invoke(prompt) == cached_response
        # async test
        await llm_cache.aupdate(
            prompt=prompt,
            llm_string=create_llm_string(llm),
            return_val=[Generation(text=cached_response)],
        )
        assert await llm.ainvoke(prompt) == cached_response
    else:
        raise ValueError(
            "The cache not set. This should never happen, as the pytest fixture "
            "`set_cache_and_teardown` always sets the cache."
        )


def test_old_sqlite_llm_caching() -> None:
    llm_cache = get_llm_cache()
    if isinstance(llm_cache, SQLAlchemyCache):
        prompt = "How are you?"
        response = "Test response"
        cached_response = "Cached test response"
        llm = FakeListLLM(responses=[response])
        items = [
            llm_cache.cache_schema(
                prompt=prompt,
                llm=create_llm_string(llm),
                response=cached_response,
                idx=0,
            )
        ]
        with Session(llm_cache.engine) as session, session.begin():
            for item in items:
                session.merge(item)
        assert llm(prompt) == cached_response


async def test_chat_model_caching() -> None:
    prompt: List[BaseMessage] = [HumanMessage(content="How are you?")]
    response = "Test response"
    cached_response = "Cached test response"
    cached_message = AIMessage(content=cached_response)
    llm = FakeListChatModel(responses=[response])
    if llm_cache := get_llm_cache():
        # sync test
        llm_cache.update(
            prompt=dumps(prompt),
            llm_string=llm._get_llm_string(),
            return_val=[ChatGeneration(message=cached_message)],
        )
        result = llm(prompt)
        assert isinstance(result, AIMessage)
        assert result.content == cached_response

        # async test
        await llm_cache.aupdate(
            prompt=dumps(prompt),
            llm_string=llm._get_llm_string(),
            return_val=[ChatGeneration(message=cached_message)],
        )
        result = await llm.ainvoke(prompt)
        assert isinstance(result, AIMessage)
        assert result.content == cached_response
    else:
        raise ValueError(
            "The cache not set. This should never happen, as the pytest fixture "
            "`set_cache_and_teardown` always sets the cache."
        )


async def test_chat_model_caching_params() -> None:
    prompt: List[BaseMessage] = [HumanMessage(content="How are you?")]
    response = "Test response"
    cached_response = "Cached test response"
    cached_message = AIMessage(content=cached_response)
    llm = FakeListChatModel(responses=[response])
    if llm_cache := get_llm_cache():
        # sync test
        llm_cache.update(
            prompt=dumps(prompt),
            llm_string=llm._get_llm_string(functions=[]),
            return_val=[ChatGeneration(message=cached_message)],
        )
        result = llm(prompt, functions=[])
        result_no_params = llm(prompt)
        assert isinstance(result, AIMessage)
        assert result.content == cached_response
        assert isinstance(result_no_params, AIMessage)
        assert result_no_params.content == response

        # async test
        await llm_cache.aupdate(
            prompt=dumps(prompt),
            llm_string=llm._get_llm_string(functions=[]),
            return_val=[ChatGeneration(message=cached_message)],
        )
        result = await llm.ainvoke(prompt, functions=[])
        result_no_params = await llm.ainvoke(prompt)
        assert isinstance(result, AIMessage)
        assert result.content == cached_response
        assert isinstance(result_no_params, AIMessage)
        assert result_no_params.content == response
    else:
        raise ValueError(
            "The cache not set. This should never happen, as the pytest fixture "
            "`set_cache_and_teardown` always sets the cache."
        )


async def test_llm_cache_clear() -> None:
    prompt = "How are you?"
    expected_response = "Test response"
    cached_response = "Cached test response"
    llm = FakeListLLM(responses=[expected_response])
    if llm_cache := get_llm_cache():
        # sync test
        llm_cache.update(
            prompt=prompt,
            llm_string=create_llm_string(llm),
            return_val=[Generation(text=cached_response)],
        )
        llm_cache.clear()
        response = llm(prompt)
        assert response == expected_response

        # async test
        await llm_cache.aupdate(
            prompt=prompt,
            llm_string=create_llm_string(llm),
            return_val=[Generation(text=cached_response)],
        )
        await llm_cache.aclear()
        response = await llm.ainvoke(prompt)
        assert response == expected_response
    else:
        raise ValueError(
            "The cache not set. This should never happen, as the pytest fixture "
            "`set_cache_and_teardown` always sets the cache."
        )


def create_llm_string(llm: Union[BaseLLM, BaseChatModel]) -> str:
    _dict: Dict = llm.dict()
    _dict["stop"] = None
    return str(sorted([(k, v) for k, v in _dict.items()]))
