"""Test caching for LLMs and ChatModels."""
from typing import Dict, Generator, List, Union

import pytest
from _pytest.fixtures import FixtureRequest
from sqlalchemy.orm import Session

import langchain
from langchain.cache import (
    InMemoryCache,
    SQLiteCache,
)
from langchain.chat_models import FakeListChatModel
from langchain.chat_models.base import BaseChatModel, dumps
from langchain.llms import FakeListLLM
from langchain.llms.base import BaseLLM
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    Generation,
    HumanMessage,
)

CACHE_DB_FILE = ".langchain.test.db"

CACHE_OPTIONS = [
    InMemoryCache(),
    SQLiteCache(CACHE_DB_FILE),
]


@pytest.fixture(autouse=True, params=CACHE_OPTIONS)
def set_cache_and_teardown(request: FixtureRequest) -> Generator[None, None, None]:
    # Will be run before each test
    cache_instance = request.param
    langchain.llm_cache = cache_instance
    if langchain.llm_cache:
        langchain.llm_cache.clear()
    else:
        raise ValueError("Cache not set. This should never happen.")

    yield

    # Will be run after each test
    if langchain.llm_cache:
        langchain.llm_cache.clear()
    else:
        raise ValueError("Cache not set. This should never happen.")


def test_llm_caching() -> None:
    prompt = "How are you?"
    response = "Test response"
    cached_response = "Cached test response"
    llm = FakeListLLM(responses=[response])
    if langchain.llm_cache:
        langchain.llm_cache.update(
            prompt=prompt,
            llm_string=create_llm_string(llm),
            return_val=[Generation(text=cached_response)],
        )
        assert llm(prompt) == cached_response
    else:
        raise ValueError(
            "The cache not set. This should never happen, as the pytest fixture "
            "`set_cache_and_teardown` always sets the cache."
        )


def test_old_sqlite_llm_caching() -> None:
    if isinstance(langchain.llm_cache, SQLiteCache):
        prompt = "How are you?"
        response = "Test response"
        cached_response = "Cached test response"
        llm = FakeListLLM(responses=[response])
        items = [
            langchain.llm_cache.cache_schema(
                prompt=prompt,
                llm=create_llm_string(llm),
                response=cached_response,
                idx=0,
            )
        ]
        with Session(langchain.llm_cache.engine) as session, session.begin():
            for item in items:
                session.merge(item)
        assert llm(prompt) == cached_response


def test_chat_model_caching() -> None:
    prompt: List[BaseMessage] = [HumanMessage(content="How are you?")]
    response = "Test response"
    cached_response = "Cached test response"
    cached_message = AIMessage(content=cached_response)
    llm = FakeListChatModel(responses=[response])
    if langchain.llm_cache:
        langchain.llm_cache.update(
            prompt=dumps(prompt),
            llm_string=create_llm_string(llm),
            return_val=[ChatGeneration(message=cached_message)],
        )
        result = llm(prompt)
        assert isinstance(result, AIMessage)
        assert llm(prompt).content == cached_response
    else:
        raise ValueError(
            "The cache not set. This should never happen, as the pytest fixture "
            "`set_cache_and_teardown` always sets the cache."
        )


def create_llm_string(llm: Union[BaseLLM, BaseChatModel]) -> str:
    _dict: Dict = llm.dict()
    _dict["stop"] = None
    return str(sorted([(k, v) for k, v in _dict.items()]))
