from typing import AsyncIterator, Iterator

from langchain_core.runnables.base import (
    coerce_to_runnable,
)
from langchain_core.runnables.config import RunnableConfig


async def test_coerce_to_runnable():
    def _foo(input: str, config: RunnableConfig) -> str:
        assert config
        return input

    assert coerce_to_runnable(_foo).invoke("foo") == "foo"

    async def _afoo(input: str, config: RunnableConfig) -> str:
        assert config
        return input

    assert (await coerce_to_runnable(_afoo).ainvoke("foo")) == "foo"

    def _foo_iter(input: Iterator[str], config: RunnableConfig) -> Iterator[str]:
        assert config
        for batch in input:
            for item in batch:
                yield f"foo_{item}"

    assert list(coerce_to_runnable(_foo_iter).stream(["bar", "baz"])) == [
        "foo_bar",
        "foo_baz",
    ]

    async def _afoo_iter(
        input: AsyncIterator[str], config: RunnableConfig
    ) -> AsyncIterator[str]:
        assert config
        async for batch in input:
            async for item in batch:
                yield f"afoo_{item}"

    async def async_generator(items):
        for item in items:
            yield item

    result = []
    async for item in coerce_to_runnable(_afoo_iter).astream(
        async_generator(["bar", "baz"])
    ):
        result.append(item)
    assert result == ["afoo_bar", "afoo_baz"]
