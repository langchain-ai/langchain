from typing import Any

from typing_extensions import override

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import FakeListLLM


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


async def test_local_cache_generate_async() -> None:
    global_cache = InMemoryCache()
    local_cache = InMemoryCache()
    try:
        set_llm_cache(global_cache)
        llm = FakeListLLM(cache=local_cache, responses=["foo", "bar"])
        output = await llm.agenerate(["foo"])
        assert output.generations[0][0].text == "foo"
        output = await llm.agenerate(["foo"])
        assert output.generations[0][0].text == "foo"
        assert global_cache._cache == {}
        assert len(local_cache._cache) == 1
    finally:
        set_llm_cache(None)


def test_local_cache_generate_sync() -> None:
    global_cache = InMemoryCache()
    local_cache = InMemoryCache()
    try:
        set_llm_cache(global_cache)
        llm = FakeListLLM(cache=local_cache, responses=["foo", "bar"])
        output = llm.generate(["foo"])
        assert output.generations[0][0].text == "foo"
        output = llm.generate(["foo"])
        assert output.generations[0][0].text == "foo"
        assert global_cache._cache == {}
        assert len(local_cache._cache) == 1
    finally:
        set_llm_cache(None)


class InMemoryCacheBad(BaseCache):
    """In-memory cache used for testing purposes."""

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache: dict[tuple[str, str], RETURN_VAL_TYPE] = {}

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        """Look up based on `prompt` and `llm_string`."""
        msg = "This code should not be triggered"
        raise NotImplementedError(msg)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on `prompt` and `llm_string`."""
        msg = "This code should not be triggered"
        raise NotImplementedError(msg)

    @override
    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        self._cache = {}


def test_no_cache_generate_sync() -> None:
    global_cache = InMemoryCacheBad()
    try:
        set_llm_cache(global_cache)
        llm = FakeListLLM(cache=False, responses=["foo", "bar"])
        output = llm.generate(["foo"])
        assert output.generations[0][0].text == "foo"
        output = llm.generate(["foo"])
        assert output.generations[0][0].text == "bar"
        assert global_cache._cache == {}
    finally:
        set_llm_cache(None)


async def test_no_cache_generate_async() -> None:
    global_cache = InMemoryCacheBad()
    try:
        set_llm_cache(global_cache)
        llm = FakeListLLM(cache=False, responses=["foo", "bar"])
        output = await llm.agenerate(["foo"])
        assert output.generations[0][0].text == "foo"
        output = await llm.agenerate(["foo"])
        assert output.generations[0][0].text == "bar"
        assert global_cache._cache == {}
    finally:
        set_llm_cache(None)


class SizedCache(BaseCache):
    """Cache that implements __len__, making an empty instance falsy.

    This exercises the `is not None` guard — a truthiness check (`if llm_cache:`)
    would skip lookups when the cache is empty, silently dropping prompts from
    both `existing_prompts` and `missing_prompts` and causing a KeyError later.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], RETURN_VAL_TYPE] = {}

    def __len__(self) -> int:
        return len(self._cache)

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        return self._cache.get((prompt, llm_string))

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        self._cache[prompt, llm_string] = return_val

    @override
    def clear(self, **kwargs: Any) -> None:
        self._cache = {}


def test_sized_cache_generate_sync() -> None:
    """generate() must not raise KeyError when cache implements __len__."""
    cache = SizedCache()
    assert len(cache) == 0  # falsy — would break under the old truthiness check

    llm = FakeListLLM(cache=cache, responses=["foo", "bar"])

    # First call: cache miss → LLM called, result stored
    output = llm.generate(["hello"])
    assert output.generations[0][0].text == "foo"
    assert len(cache) == 1

    # Second call: cache hit → same result returned, LLM NOT called again
    output = llm.generate(["hello"])
    assert output.generations[0][0].text == "foo"
    assert len(cache) == 1  # still 1 — no new entry


async def test_sized_cache_generate_async() -> None:
    """agenerate() must not raise KeyError when cache implements __len__."""
    cache = SizedCache()
    assert len(cache) == 0

    llm = FakeListLLM(cache=cache, responses=["foo", "bar"])

    output = await llm.agenerate(["hello"])
    assert output.generations[0][0].text == "foo"
    assert len(cache) == 1

    output = await llm.agenerate(["hello"])
    assert output.generations[0][0].text == "foo"
    assert len(cache) == 1
