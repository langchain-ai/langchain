from typing import Any

from langchain_core.utils.curry import curry


def test_curry() -> None:
    def test_fn(a: str, b: str):
        return a + b

    curried = curry(test_fn, a="hey")
    assert curried(b=" you") == "hey you"


def test_curry_with_kwargs_values() -> None:
    def test_fn(a: str, b: str, **kwargs: Any):
        return a + b + kwargs["c"]

    curried = curry(test_fn, c=" you you")
    assert curried(a="hey", b=" hey") == "hey hey you you"


def test_noop_curry() -> None:
    def test_fn(a: str, b: str):
        return a + b

    curried = curry(test_fn)
    assert curried(a="bye", b=" you") == "bye you"


async def test_async_curry() -> None:
    async def test_fn(a: str, b: str):
        return a + b

    curried = curry(test_fn, a="hey")
    assert await curried(b=" you") == "hey you"


async def test_async_curry_with_kwargs_values() -> None:
    async def test_fn(a: str, b: str, **kwargs: Any):
        return a + b + kwargs["c"]

    curried = curry(test_fn, c=" you you")
    assert await curried(a="hey", b=" hey") == "hey hey you you"


async def test_noop_async_curry() -> None:
    async def test_fn(a: str, b: str):
        return a + b

    curried = curry(test_fn)
    assert await curried(a="bye", b=" you") == "bye you"
