"""Test caching for LLMs and ChatModels."""

import sqlite3
from typing import Dict, Generator, List, Union

import pytest
from _pytest.fixtures import FixtureRequest
from langchain_core.caches import InMemoryCache
from langchain_core.language_models import FakeListChatModel, FakeListLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.load import dumps
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration
from sqlalchemy import Column, Integer, Sequence, String, create_engine
from sqlalchemy.orm import Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from langchain.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation, LLMResult

from langchain_community.cache import SQLAlchemyCache
from tests.unit_tests.llms.fake_llm import FakeLLM


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
        assert llm.invoke(prompt) == cached_response


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
        result = llm.invoke(prompt)
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
        result = llm.invoke(prompt, functions=[])
        result_no_params = llm.invoke(prompt)
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
        response = llm.invoke(prompt)
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


def test_sql_alchemy_cache() -> None:
    """Test custom_caching behavior."""
    Base = declarative_base()

    class FulltextLLMCache(Base):  # type: ignore
        """Postgres table for fulltext-indexed LLM Cache."""

        __tablename__ = "llm_cache_fulltext"
        id = Column(Integer, Sequence("cache_id"), primary_key=True)
        prompt = Column(String, nullable=False)
        llm = Column(String, nullable=False)
        idx = Column(Integer)
        response = Column(String)

    engine = create_engine("sqlite://")

    from langchain_community.cache import SQLAlchemyCache

    set_llm_cache(SQLAlchemyCache(engine, FulltextLLMCache))
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(["foo", "bar", "foo"])
    expected_cache_output = [Generation(text="foo")]
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == expected_cache_output
    set_llm_cache(None)
    expected_generations = [
        [Generation(text="fizz")],
        [Generation(text="foo")],
        [Generation(text="fizz")],
    ]
    expected_output = LLMResult(
        generations=expected_generations,
        llm_output=None,
    )
    assert output == expected_output
