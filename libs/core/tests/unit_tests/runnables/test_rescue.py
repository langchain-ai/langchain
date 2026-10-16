from typing import Any

import pytest

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableGenerator, RunnableLambda


def _raise_value_error(_: Any) -> Any:
    msg = "test error"
    raise ValueError(msg)


def test_rescue_with_fallback() -> None:
    runnable = RunnableLambda(_raise_value_error).with_rescue(fallback="recovered")
    assert runnable.invoke("hello") == "recovered"


def test_rescue_with_none_fallback() -> None:
    runnable = RunnableLambda(_raise_value_error).with_rescue(fallback=None)
    assert runnable.invoke("hello") is None


def test_rescue_with_handler() -> None:
    runnable = RunnableLambda(_raise_value_error).with_rescue(
        handler=lambda error, input_data: {"error": str(error), "input": input_data}
    )
    assert runnable.invoke("hello") == {"error": "test error", "input": "hello"}


def test_rescue_exception_filter() -> None:
    runnable = RunnableLambda(_raise_value_error).with_rescue(
        fallback="recovered", exceptions=(TypeError,)
    )
    with pytest.raises(ValueError, match="test error"):
        runnable.invoke("hello")


def test_rescue_inside_pipeline() -> None:
    pipeline = (
        RunnableLambda(lambda x: x + 1)
        | RunnableLambda(_raise_value_error).with_rescue(fallback=5)
        | RunnableLambda(lambda x: x * 2)
    )
    assert pipeline.invoke(1) == 10


def test_rescue_without_recovery() -> None:
    runnable = RunnableLambda(_raise_value_error).with_rescue()
    with pytest.raises(ValueError, match="test error"):
        runnable.invoke("hello")


async def test_rescue_async() -> None:
    runnable = RunnableLambda(_raise_value_error).with_rescue(fallback="recovered")
    assert await runnable.ainvoke("hello") == "recovered"


async def test_rescue_async_handler() -> None:
    async def handler(error: Exception, input_data: Any) -> dict[str, Any]:
        return {"error": str(error), "input": input_data}

    runnable = RunnableLambda(_raise_value_error).with_rescue(async_handler=handler)
    assert await runnable.ainvoke("hello") == {"error": "test error", "input": "hello"}


class RescueCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.started = 0
        self.ended = 0
        self.errors = 0

    def on_chain_start(
        self,
        _serialized: dict[str, Any] | None,
        _inputs: dict[str, Any],
        **_kwargs: Any,
    ) -> None:
        self.started += 1

    def on_chain_end(
        self,
        _outputs: dict[str, Any],
        **_kwargs: Any,
    ) -> None:
        self.ended += 1

    def on_chain_error(
        self,
        _error: BaseException,
        **_kwargs: Any,
    ) -> None:
        self.errors += 1


def test_rescue_callback_propagation() -> None:
    callback = RescueCallbackHandler()
    runnable = RunnableLambda(_raise_value_error).with_rescue(fallback="recovered")

    assert runnable.invoke("hello", config={"callbacks": [callback]}) == "recovered"
    assert callback.started == 2
    assert callback.ended == 1
    assert callback.errors == 1


async def test_rescue_async_callback_propagation() -> None:
    callback = RescueCallbackHandler()
    runnable = RunnableLambda(_raise_value_error).with_rescue(fallback="recovered")

    result = await runnable.ainvoke(
        "hello",
        config={"callbacks": [callback]},
    )

    assert result == "recovered"
    assert callback.started == 2
    assert callback.ended == 1
    assert callback.errors == 1


def test_rescue_handler_raises_error() -> None:
    def failing_handler(
        _error: Exception,
        _input_data: Any,
    ) -> Any:
        msg = "recovery failed"
        raise RuntimeError(msg)

    runnable = RunnableLambda(_raise_value_error).with_rescue(handler=failing_handler)

    with pytest.raises(RuntimeError, match="recovery failed"):
        runnable.invoke("hello")


async def test_rescue_async_exception_filter() -> None:
    runnable = RunnableLambda(_raise_value_error).with_rescue(
        fallback="recovered",
        exceptions=(TypeError,),
    )

    with pytest.raises(ValueError, match="test error"):
        await runnable.ainvoke("hello")


async def test_rescue_astream() -> None:
    runnable = RunnableLambda(_raise_value_error).with_rescue(fallback="recovered")

    assert [chunk async for chunk in runnable.astream("hello")] == ["recovered"]


def test_rescue_stream() -> None:
    runnable = RunnableLambda(_raise_value_error).with_rescue(fallback="recovered")

    assert list(runnable.stream("hello")) == ["recovered"]


def test_rescue_stream_error_after_first_chunk() -> None:
    def failing_stream(_input: Any) -> Any:
        yield "first"
        msg = "stream failed"
        raise ValueError(msg)

    runnable = RunnableGenerator(failing_stream).with_rescue(fallback="recovered")

    stream = runnable.stream("hello")
    assert next(stream) == "first"

    with pytest.raises(ValueError, match="stream failed"):
        next(stream)


async def test_rescue_astream_error_after_first_chunk() -> None:
    async def failing_stream(_input: Any) -> Any:
        yield "first"
        msg = "async stream failed"
        raise ValueError(msg)

    runnable = RunnableGenerator(failing_stream).with_rescue(fallback="recovered")

    stream = runnable.astream("hello")
    assert await anext(stream) == "first"

    with pytest.raises(ValueError, match="async stream failed"):
        await anext(stream)
