"""Module tests interaction of chat model with caching abstraction.."""

from typing import Any, Optional

import pytest

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.chat_models import _cleanup_llm_representation
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration


class InMemoryCache(BaseCache):
    """In-memory cache used for testing purposes."""

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache: dict[tuple[str, str], RETURN_VAL_TYPE] = {}

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        return self._cache.get((prompt, llm_string), None)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        self._cache[(prompt, llm_string)] = return_val

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        self._cache = {}


def test_local_cache_sync() -> None:
    """Test that the local cache is being populated but not the global one."""
    global_cache = InMemoryCache()
    local_cache = InMemoryCache()
    try:
        set_llm_cache(global_cache)
        chat_model = FakeListChatModel(
            cache=local_cache, responses=["hello", "goodbye"]
        )
        assert chat_model.invoke("How are you?").content == "hello"
        # If the cache works we should get the same response since
        # the prompt is the same
        assert chat_model.invoke("How are you?").content == "hello"
        # The global cache should be empty
        assert global_cache._cache == {}
        # The local cache should be populated
        assert len(local_cache._cache) == 1
        llm_result = list(local_cache._cache.values())
        chat_generation = llm_result[0][0]
        assert isinstance(chat_generation, ChatGeneration)
        assert chat_generation.message.content == "hello"
        # Verify that another prompt will trigger the call to the model
        assert chat_model.invoke("meow?").content == "goodbye"
        # The global cache should be empty
        assert global_cache._cache == {}
        # The local cache should be populated
        assert len(local_cache._cache) == 2
    finally:
        set_llm_cache(None)


async def test_local_cache_async() -> None:
    # Use MockCache as the cache
    global_cache = InMemoryCache()
    local_cache = InMemoryCache()
    try:
        set_llm_cache(global_cache)
        chat_model = FakeListChatModel(
            cache=local_cache, responses=["hello", "goodbye"]
        )
        assert (await chat_model.ainvoke("How are you?")).content == "hello"
        # If the cache works we should get the same response since
        # the prompt is the same
        assert (await chat_model.ainvoke("How are you?")).content == "hello"
        # The global cache should be empty
        assert global_cache._cache == {}
        # The local cache should be populated
        assert len(local_cache._cache) == 1
        llm_result = list(local_cache._cache.values())
        chat_generation = llm_result[0][0]
        assert isinstance(chat_generation, ChatGeneration)
        assert chat_generation.message.content == "hello"
        # Verify that another prompt will trigger the call to the model
        assert chat_model.invoke("meow?").content == "goodbye"
        # The global cache should be empty
        assert global_cache._cache == {}
        # The local cache should be populated
        assert len(local_cache._cache) == 2
    finally:
        set_llm_cache(None)


def test_global_cache_sync() -> None:
    """Test that the global cache gets populated when cache = True."""
    global_cache = InMemoryCache()
    try:
        set_llm_cache(global_cache)
        chat_model = FakeListChatModel(
            cache=True, responses=["hello", "goodbye", "meow", "woof"]
        )
        assert (chat_model.invoke("How are you?")).content == "hello"
        # If the cache works we should get the same response since
        # the prompt is the same
        assert (chat_model.invoke("How are you?")).content == "hello"
        # The global cache should be populated
        assert len(global_cache._cache) == 1
        llm_result = list(global_cache._cache.values())
        chat_generation = llm_result[0][0]
        assert isinstance(chat_generation, ChatGeneration)
        assert chat_generation.message.content == "hello"
        # Verify that another prompt will trigger the call to the model
        assert chat_model.invoke("nice").content == "goodbye"
        # The local cache should be populated
        assert len(global_cache._cache) == 2
    finally:
        set_llm_cache(None)


async def test_global_cache_async() -> None:
    """Test that the global cache gets populated when cache = True."""
    global_cache = InMemoryCache()
    try:
        set_llm_cache(global_cache)
        chat_model = FakeListChatModel(
            cache=True, responses=["hello", "goodbye", "meow", "woof"]
        )
        assert (await chat_model.ainvoke("How are you?")).content == "hello"
        # If the cache works we should get the same response since
        # the prompt is the same
        assert (await chat_model.ainvoke("How are you?")).content == "hello"
        # The global cache should be populated
        assert len(global_cache._cache) == 1
        llm_result = list(global_cache._cache.values())
        chat_generation = llm_result[0][0]
        assert isinstance(chat_generation, ChatGeneration)
        assert chat_generation.message.content == "hello"
        # Verify that another prompt will trigger the call to the model
        assert chat_model.invoke("nice").content == "goodbye"
        # The local cache should be populated
        assert len(global_cache._cache) == 2
    finally:
        set_llm_cache(None)


def test_no_cache_sync() -> None:
    global_cache = InMemoryCache()
    try:
        set_llm_cache(global_cache)
        chat_model = FakeListChatModel(
            cache=False, responses=["hello", "goodbye"]
        )  # Set cache=False
        assert (chat_model.invoke("How are you?")).content == "hello"
        # The global cache should not be populated since cache=False
        # so we should get the second response
        assert (chat_model.invoke("How are you?")).content == "goodbye"
        # The global cache should not be populated since cache=False
        assert len(global_cache._cache) == 0
    finally:
        set_llm_cache(None)


async def test_no_cache_async() -> None:
    global_cache = InMemoryCache()
    try:
        set_llm_cache(global_cache)
        chat_model = FakeListChatModel(
            cache=False, responses=["hello", "goodbye"]
        )  # Set cache=False
        assert (await chat_model.ainvoke("How are you?")).content == "hello"
        # The global cache should not be populated since cache=False
        # so we should get the second response
        assert (await chat_model.ainvoke("How are you?")).content == "goodbye"
        # The global cache should not be populated since cache=False
        assert len(global_cache._cache) == 0
    finally:
        set_llm_cache(None)


async def test_global_cache_abatch() -> None:
    global_cache = InMemoryCache()
    try:
        set_llm_cache(global_cache)
        chat_model = FakeListChatModel(
            cache=True, responses=["hello", "goodbye", "meow", "woof"]
        )
        results = await chat_model.abatch(["first prompt", "second prompt"])
        assert results[0].content == "hello"
        assert results[1].content == "goodbye"

        # Now try with the same prompt
        results = await chat_model.abatch(["first prompt", "first prompt"])
        assert results[0].content == "hello"
        assert results[1].content == "hello"

        global_cache = InMemoryCache()
        set_llm_cache(global_cache)
        assert global_cache._cache == {}
        results = await chat_model.abatch(["prompt", "prompt"])

        assert results[0].content == "meow"
        assert results[1].content == "meow"
    finally:
        set_llm_cache(None)


def test_global_cache_batch() -> None:
    global_cache = InMemoryCache()
    try:
        set_llm_cache(global_cache)
        chat_model = FakeListChatModel(
            cache=True, responses=["hello", "goodbye", "meow", "woof"]
        )
        results = chat_model.batch(["first prompt", "second prompt"])
        # These may be in any order
        assert {results[0].content, results[1].content} == {"hello", "goodbye"}

        # Now try with the same prompt
        results = chat_model.batch(["first prompt", "first prompt"])
        # These could be either "hello" or "goodbye" and should be identical
        assert results[0].content == results[1].content
        assert {results[0].content, results[1].content}.issubset({"hello", "goodbye"})

        # RACE CONDITION -- note behavior is different from async
        # Now, reset cache and test the race condition
        # For now we just hard-code the result, if this changes
        # we can investigate further
        global_cache = InMemoryCache()
        set_llm_cache(global_cache)
        assert global_cache._cache == {}
        results = chat_model.batch(
            [
                "prompt",
                "prompt",
            ]
        )
        assert {results[0].content, results[1].content} == {"meow"}
    finally:
        set_llm_cache(None)


@pytest.mark.xfail(reason="Abstraction does not support caching for streaming yet.")
def test_global_cache_stream() -> None:
    """Test streaming."""
    global_cache = InMemoryCache()
    try:
        set_llm_cache(global_cache)
        messages = [
            AIMessage(content="hello world"),
            AIMessage(content="goodbye world"),
        ]
        model = GenericFakeChatModel(messages=iter(messages), cache=True)
        chunks = list(model.stream("some input"))
        assert len(chunks) == 3
        # Assert that streaming information gets cached
        assert global_cache._cache != {}
    finally:
        set_llm_cache(None)


class CustomChat(GenericFakeChatModel):
    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


async def test_can_swap_caches() -> None:
    """Test that we can use a different cache object.

    This test verifies that when we fetch the llm_string representation
    of the chat model, we can swap the cache object and still get the same
    result.
    """
    cache = InMemoryCache()
    chat_model = CustomChat(cache=cache, messages=iter(["hello"]))
    result = await chat_model.ainvoke("foo")
    assert result.content == "hello"

    new_cache = InMemoryCache()
    new_cache._cache = cache._cache.copy()

    # Confirm that we get a cache hit!
    chat_model = CustomChat(cache=new_cache, messages=iter(["goodbye"]))
    result = await chat_model.ainvoke("foo")
    assert result.content == "hello"


def test_llm_representation_for_serializable() -> None:
    """Test that the llm representation of a serializable chat model is correct."""
    cache = InMemoryCache()
    chat = CustomChat(cache=cache, messages=iter([]))
    assert chat._get_llm_string() == (
        '{"id": ["tests", "unit_tests", "language_models", "chat_models", '
        '"test_cache", "CustomChat"], "kwargs": {"messages": {"id": '
        '["builtins", "list_iterator"], "lc": 1, "type": "not_implemented"}}, "lc": '
        '1, "name": "CustomChat", "type": "constructor"}---[(\'stop\', None)]'
    )


def test_cleanup_serialized() -> None:
    cleanup_serialized = {
        "lc": 1,
        "type": "constructor",
        "id": [
            "tests",
            "unit_tests",
            "language_models",
            "chat_models",
            "test_cache",
            "CustomChat",
        ],
        "kwargs": {
            "messages": {
                "lc": 1,
                "type": "not_implemented",
                "id": ["builtins", "list_iterator"],
                "repr": "<list_iterator object at 0x79ff437f8d30>",
            },
        },
        "name": "CustomChat",
        "graph": {
            "nodes": [
                {"id": 0, "type": "schema", "data": "CustomChatInput"},
                {
                    "id": 1,
                    "type": "runnable",
                    "data": {
                        "id": [
                            "tests",
                            "unit_tests",
                            "language_models",
                            "chat_models",
                            "test_cache",
                            "CustomChat",
                        ],
                        "name": "CustomChat",
                    },
                },
                {"id": 2, "type": "schema", "data": "CustomChatOutput"},
            ],
            "edges": [{"source": 0, "target": 1}, {"source": 1, "target": 2}],
        },
    }
    _cleanup_llm_representation(cleanup_serialized, 1)
    assert cleanup_serialized == {
        "id": [
            "tests",
            "unit_tests",
            "language_models",
            "chat_models",
            "test_cache",
            "CustomChat",
        ],
        "kwargs": {
            "messages": {
                "id": ["builtins", "list_iterator"],
                "lc": 1,
                "type": "not_implemented",
            },
        },
        "lc": 1,
        "name": "CustomChat",
        "type": "constructor",
    }
