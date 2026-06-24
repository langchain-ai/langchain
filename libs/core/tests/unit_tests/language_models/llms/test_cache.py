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


class SizedInMemoryCache(InMemoryCache):
    """In-memory cache that reports its size via ``__len__``.

    Many real cache implementations expose their entry count this way, which makes
    an empty cache falsy. Used to guard against truthiness checks being used where
    an identity check is intended.
    """

    def __len__(self) -> int:
        return len(self._cache)


async def test_local_cache_generate_async_sized_cache() -> None:
    """A cache that defines ``__len__`` must work with ``agenerate``.

    Regression test: ``aget_prompts``/``aupdate_cache`` previously used a
    truthiness check, so an empty cache reporting ``len() == 0`` was treated as
    absent. The prompt was added to neither the existing nor the missing set,
    which raised ``KeyError`` when building the result.
    """
    local_cache = SizedInMemoryCache()
    assert len(local_cache) == 0  # empty -> falsy
    llm = FakeListLLM(cache=local_cache, responses=["foo", "bar"])
    output = await llm.agenerate(["foo"])
    assert output.generations[0][0].text == "foo"
    # Second call must be served from the cache, not generate "bar".
    output = await llm.agenerate(["foo"])
    assert output.generations[0][0].text == "foo"
    assert len(local_cache._cache) == 1


def test_local_cache_generate_sync_sized_cache() -> None:
    """Sync counterpart of `test_local_cache_generate_async_sized_cache`."""
    local_cache = SizedInMemoryCache()
    assert len(local_cache) == 0  # empty -> falsy
    llm = FakeListLLM(cache=local_cache, responses=["foo", "bar"])
    output = llm.generate(["foo"])
    assert output.generations[0][0].text == "foo"
    output = llm.generate(["foo"])
    assert output.generations[0][0].text == "foo"
    assert len(local_cache._cache) == 1


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
