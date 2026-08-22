"""Exception propagation out of `RunnableSequence.astream`.

#38074 reported a silent EOF when a sequence step raised inside `astream`.
Current `master` already re-raises. These tests lock that contract and cover
the two cleanup holes on the same async stream error path.
"""

from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

import pytest

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableGenerator, RunnableLambda, RunnableParallel

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain_core.runnables.base import Runnable


class BoomError(Exception):
    """Sentinel raised by the fixtures in this module."""


class ErrorRecorder(BaseCallbackHandler):
    """Collects `on_chain_error` payloads."""

    def __init__(self) -> None:
        self.errors: list[BaseException] = []

    def on_chain_error(self, error: BaseException, **_kwargs: Any) -> None:
        self.errors.append(error)


def _boom(_: object) -> object:
    msg = "step"
    raise BoomError(msg)


async def _agen_mid(inputs: AsyncIterator[str]) -> AsyncIterator[str]:
    async for _x in inputs:
        yield "pre"
        msg = "mid-stream"
        raise BoomError(msg)


async def _drain(chain: Runnable[Any, Any], value: object = "hi") -> list[Any]:
    return [chunk async for chunk in chain.astream(value)]


@pytest.mark.parametrize(
    "chain",
    [
        RunnableLambda(lambda x: x) | RunnableLambda(_boom),
        RunnableLambda(_boom) | RunnableLambda(lambda x: x),
        RunnableLambda(lambda x: x)
        | RunnableLambda(_boom)
        | RunnableLambda(lambda x: x),
    ],
    ids=["last-step", "first-step", "middle-step"],
)
async def test_sequence_astream_raises_when_a_step_raises(
    chain: Runnable[Any, Any],
) -> None:
    """A raising step must come out of `async for`, not end the stream."""
    with pytest.raises(BoomError, match="step"):
        await _drain(chain)


async def test_sequence_astream_raises_after_a_streamed_chunk() -> None:
    """Chunks yielded before the error stay visible; the error still surfaces."""
    chain: Runnable[str, str] = RunnableLambda(lambda x: x) | RunnableGenerator(
        _agen_mid
    )
    chunks: list[str] = []

    async def _consume() -> None:
        async for chunk in chain.astream("hi"):
            chunks.append(chunk)  # noqa: PERF401

    with pytest.raises(BoomError, match="mid-stream"):
        await _consume()
    assert chunks == ["pre"]


async def test_sequence_stream_and_ainvoke_raise_the_same_error() -> None:
    """`stream` and `ainvoke` already raised; keep them aligned with `astream`."""
    chain: Runnable[str, object] = RunnableLambda(lambda x: x) | RunnableLambda(_boom)
    with pytest.raises(BoomError, match="step"):
        list(chain.stream("hi"))
    with pytest.raises(BoomError, match="step"):
        await chain.ainvoke("hi")


async def test_sequence_astream_records_on_chain_error() -> None:
    """The run should be marked failed, not closed as a successful empty stream."""
    handler = ErrorRecorder()
    chain: Runnable[str, object] = RunnableLambda(lambda x: x) | RunnableLambda(_boom)
    with pytest.raises(BoomError, match="step"):
        await _drain(chain.with_config(callbacks=[handler]))
    assert any(isinstance(err, BoomError) for err in handler.errors)


async def test_parallel_astream_raises_without_orphaning_sibling_tasks() -> None:
    """A raising parallel branch must not leave `StopAsyncIteration` unretrieved."""
    leaked: list[BaseException] = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()

    def _handler(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        if (exc := context.get("exception")) is not None:
            leaked.append(exc)

    loop.set_exception_handler(_handler)
    try:
        chain = RunnableParallel(
            ok=RunnableLambda(lambda x: x),
            bad=RunnableLambda(_boom),
        )
        with pytest.raises(BoomError, match="step"):
            await _drain(chain)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous)

    assert leaked == []


async def test_astream_setup_error_is_not_hidden_by_aclose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If stream setup fails, `finally` must not replace it with `NameError`."""

    class _BoomContext(AbstractContextManager[None]):
        def __enter__(self) -> None:
            msg = "setup failed"
            raise RuntimeError(msg)

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "langchain_core.runnables.base.set_config_context",
        lambda *_args: _BoomContext(),
    )
    chain: Runnable[str, object] = RunnableLambda(lambda x: x) | RunnableLambda(
        lambda x: x
    )
    with pytest.raises(RuntimeError, match="setup failed"):
        await _drain(chain)
