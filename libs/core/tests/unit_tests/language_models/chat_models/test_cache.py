"""Module tests interaction of chat model with caching abstraction.."""

from typing import Any

import pytest
from typing_extensions import override

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.chat_models import _cleanup_llm_representation
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.load import dumps
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.outputs.chat_result import ChatResult


class InMemoryCache(BaseCache):
    """In-memory cache used for testing purposes."""

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache: dict[tuple[str, str], RETURN_VAL_TYPE] = {}

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        """Look up based on `prompt` and `llm_string`."""
        return self._cache.get((prompt, llm_string), None)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on `prompt` and `llm_string`."""
        self._cache[prompt, llm_string] = return_val

    @override
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


def test_cache_with_generation_objects() -> None:
    """Test that cache can handle Generation objects instead of ChatGeneration objects.

    This test reproduces a bug where cache returns Generation objects
    but ChatResult expects ChatGeneration objects, causing validation errors.

    See #22389 for more info.

    """
    cache = InMemoryCache()

    # Create a simple fake chat model that we can control
    class SimpleFakeChat:
        """Simple fake chat model for testing."""

        def __init__(self, cache: BaseCache) -> None:
            self.cache = cache
            self.response = "hello"

        def _get_llm_string(self) -> str:
            return "test_llm_string"

        def generate_response(self, prompt: str) -> ChatResult:
            """Simulate the cache lookup and generation logic."""
            llm_string = self._get_llm_string()
            prompt_str = dumps([prompt])

            # Check cache first
            cache_val = self.cache.lookup(prompt_str, llm_string)
            if cache_val:
                # This is where our fix should work
                converted_generations = []
                for gen in cache_val:
                    if isinstance(gen, Generation) and not isinstance(
                        gen, ChatGeneration
                    ):
                        # Convert Generation to ChatGeneration by creating an AIMessage
                        chat_gen = ChatGeneration(
                            message=AIMessage(content=gen.text),
                            generation_info=gen.generation_info,
                        )
                        converted_generations.append(chat_gen)
                    else:
                        converted_generations.append(gen)
                return ChatResult(generations=converted_generations)

            # Generate new response
            chat_gen = ChatGeneration(
                message=AIMessage(content=self.response), generation_info={}
            )
            result = ChatResult(generations=[chat_gen])

            # Store in cache
            self.cache.update(prompt_str, llm_string, result.generations)
            return result

    model = SimpleFakeChat(cache)

    # First call - normal operation
    result1 = model.generate_response("test prompt")
    assert result1.generations[0].message.content == "hello"

    # Manually corrupt the cache by replacing ChatGeneration with Generation
    cache_key = next(iter(cache._cache.keys()))
    cached_chat_generations = cache._cache[cache_key]

    # Replace with Generation objects (missing message field)
    corrupted_generations = [
        Generation(
            text=gen.text,
            generation_info=gen.generation_info,
            type="Generation",  # This is the key - wrong type
        )
        for gen in cached_chat_generations
    ]
    cache._cache[cache_key] = corrupted_generations

    # Second call should handle the Generation objects gracefully
    result2 = model.generate_response("test prompt")
    assert result2.generations[0].message.content == "hello"
    assert isinstance(result2.generations[0], ChatGeneration)


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


def test_token_costs_are_zeroed_out() -> None:
    # We zero-out token costs for cache hits
    local_cache = InMemoryCache()
    messages = [
        AIMessage(
            content="Hello, how are you?",
            usage_metadata={"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
        ),
    ]
    model = GenericFakeChatModel(messages=iter(messages), cache=local_cache)
    first_response = model.invoke("Hello")
    assert isinstance(first_response, AIMessage)
    assert first_response.usage_metadata

    second_response = model.invoke("Hello")
    assert isinstance(second_response, AIMessage)
    assert second_response.usage_metadata
    assert second_response.usage_metadata["total_cost"] == 0  # type: ignore[typeddict-item]


def test_cache_key_ignores_message_id_sync() -> None:
    """Test that message IDs are stripped from cache keys (sync).

    Functionally identical messages with different IDs should produce
    the same cache key and result in cache hits.
    """
    local_cache = InMemoryCache()
    model = FakeListChatModel(cache=local_cache, responses=["hello", "goodbye"])

    # First call with a message that has an ID
    msg_with_id_1 = HumanMessage(content="How are you?", id="unique-id-1")
    result_1 = model.invoke([msg_with_id_1])
    assert result_1.content == "hello"

    # Second call with the same content but different ID should hit cache
    msg_with_id_2 = HumanMessage(content="How are you?", id="unique-id-2")
    result_2 = model.invoke([msg_with_id_2])
    # Should get cached response, not "goodbye"
    assert result_2.content == "hello"

    # Third call with no ID should also hit cache
    msg_no_id = HumanMessage(content="How are you?")
    result_3 = model.invoke([msg_no_id])
    assert result_3.content == "hello"

    # Verify only one cache entry exists
    assert len(local_cache._cache) == 1


async def test_cache_key_ignores_message_id_async() -> None:
    """Test that message IDs are stripped from cache keys (async).

    Functionally identical messages with different IDs should produce
    the same cache key and result in cache hits.
    """
    local_cache = InMemoryCache()
    model = FakeListChatModel(cache=local_cache, responses=["hello", "goodbye"])

    # First call with a message that has an ID
    msg_with_id_1 = HumanMessage(content="How are you?", id="unique-id-1")
    result_1 = await model.ainvoke([msg_with_id_1])
    assert result_1.content == "hello"

    # Second call with the same content but different ID should hit cache
    msg_with_id_2 = HumanMessage(content="How are you?", id="unique-id-2")
    result_2 = await model.ainvoke([msg_with_id_2])
    # Should get cached response, not "goodbye"
    assert result_2.content == "hello"

    # Third call with no ID should also hit cache
    msg_no_id = HumanMessage(content="How are you?")
    result_3 = await model.ainvoke([msg_no_id])
    assert result_3.content == "hello"

    # Verify only one cache entry exists
    assert len(local_cache._cache) == 1
