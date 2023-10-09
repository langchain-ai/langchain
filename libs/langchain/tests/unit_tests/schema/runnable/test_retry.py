from typing import AsyncIterator, Iterator, Union
from langchain.schema.runnable.utils import RunnableStreamResetMarker, aadd, add

import pytest
from pytest_mock import MockerFixture
from langchain.schema.runnable.base import Runnable, RunnableGenerator, RunnableLambda


def test_retrying(mocker: MockerFixture) -> None:
    def _lambda(x: int) -> Union[int, Runnable]:
        if x == 1:
            raise ValueError("x is 1")
        elif x == 2:
            raise RuntimeError("x is 2")
        else:
            return x

    _lambda_mock = mocker.Mock(side_effect=_lambda)
    runnable = RunnableLambda(_lambda_mock)

    with pytest.raises(ValueError):
        runnable.invoke(1)

    assert _lambda_mock.call_count == 1
    _lambda_mock.reset_mock()

    with pytest.raises(ValueError):
        runnable.with_retry(
            stop_after_attempt=2,
            retry_if_exception_type=(ValueError,),
        ).invoke(1)

    assert _lambda_mock.call_count == 2  # retried
    _lambda_mock.reset_mock()

    with pytest.raises(RuntimeError):
        runnable.with_retry(
            stop_after_attempt=2,
            wait_exponential_jitter=False,
            retry_if_exception_type=(ValueError,),
        ).invoke(2)

    assert _lambda_mock.call_count == 1  # did not retry
    _lambda_mock.reset_mock()

    with pytest.raises(ValueError):
        runnable.with_retry(
            stop_after_attempt=2,
            wait_exponential_jitter=False,
            retry_if_exception_type=(ValueError,),
        ).batch([1, 2, 0])

    # 3rd input isn't retried because it succeeded
    assert _lambda_mock.call_count == 3 + 2
    _lambda_mock.reset_mock()

    output = runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    ).batch([1, 2, 0], return_exceptions=True)

    # 3rd input isn't retried because it succeeded
    assert _lambda_mock.call_count == 3 + 2
    assert len(output) == 3
    assert isinstance(output[0], ValueError)
    assert isinstance(output[1], RuntimeError)
    assert output[2] == 0
    _lambda_mock.reset_mock()


def test_retrying_stream(mocker: MockerFixture) -> None:
    _lambda_mock = mocker.Mock()
    attempt = 0

    def _lambda(xiter: Iterator[int]) -> Iterator[int]:
        nonlocal attempt

        _lambda_mock()
        yield 1
        if attempt == 0:
            attempt += 1
            raise ValueError("x is 1")
        else:
            yield add(xiter)

    runnable = RunnableGenerator(_lambda)

    with pytest.raises(ValueError):
        [chunk for chunk in runnable.stream(1)]

    assert _lambda_mock.call_count == 1
    _lambda_mock.reset_mock()
    attempt = 0

    chunks = [
        c
        for c in runnable.with_retry(
            stop_after_attempt=2,
            retry_if_exception_type=(ValueError,),
        ).stream(3)
    ]

    assert chunks == [1, RunnableStreamResetMarker(), 1, 3]
    assert _lambda_mock.call_count == 2  # retried
    _lambda_mock.reset_mock()
    attempt = 0

    def add_10(input: Iterator[int]) -> Iterator[int]:
        for chunk in input:
            yield chunk - 10

    seq = (
        runnable.with_retry(
            stop_after_attempt=2,
            retry_if_exception_type=(ValueError,),
        )
        | add_10
    )

    chunks = [c for c in seq.stream(3)]

    assert chunks == [11, RunnableStreamResetMarker(), 11, 13]


@pytest.mark.asyncio
async def test_async_retrying(mocker: MockerFixture) -> None:
    def _lambda(x: int) -> Union[int, Runnable]:
        if x == 1:
            raise ValueError("x is 1")
        elif x == 2:
            raise RuntimeError("x is 2")
        else:
            return x

    _lambda_mock = mocker.Mock(side_effect=_lambda)
    runnable = RunnableLambda(_lambda_mock)

    with pytest.raises(ValueError):
        await runnable.ainvoke(1)

    assert _lambda_mock.call_count == 1
    _lambda_mock.reset_mock()

    with pytest.raises(ValueError):
        await runnable.with_retry(
            stop_after_attempt=2,
            wait_exponential_jitter=False,
            retry_if_exception_type=(ValueError, KeyError),
        ).ainvoke(1)

    assert _lambda_mock.call_count == 2  # retried
    _lambda_mock.reset_mock()

    with pytest.raises(RuntimeError):
        await runnable.with_retry(
            stop_after_attempt=2,
            wait_exponential_jitter=False,
            retry_if_exception_type=(ValueError,),
        ).ainvoke(2)

    assert _lambda_mock.call_count == 1  # did not retry
    _lambda_mock.reset_mock()

    with pytest.raises(ValueError):
        await runnable.with_retry(
            stop_after_attempt=2,
            wait_exponential_jitter=False,
            retry_if_exception_type=(ValueError,),
        ).abatch([1, 2, 0])

    # 3rd input isn't retried because it succeeded
    assert _lambda_mock.call_count == 3 + 2
    _lambda_mock.reset_mock()

    output = await runnable.with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=False,
        retry_if_exception_type=(ValueError,),
    ).abatch([1, 2, 0], return_exceptions=True)

    # 3rd input isn't retried because it succeeded
    assert _lambda_mock.call_count == 3 + 2
    assert len(output) == 3
    assert isinstance(output[0], ValueError)
    assert isinstance(output[1], RuntimeError)
    assert output[2] == 0
    _lambda_mock.reset_mock()


@pytest.mark.asyncio
async def test_retrying_astream(mocker: MockerFixture) -> None:
    _lambda_mock = mocker.Mock()
    attempt = 0

    async def _lambda(xiter: AsyncIterator[int]) -> AsyncIterator[int]:
        nonlocal attempt

        _lambda_mock()
        yield 1
        if attempt == 0:
            attempt += 1
            raise ValueError("x is 1")
        else:
            yield await aadd(xiter)

    runnable = RunnableGenerator(_lambda)

    with pytest.raises(ValueError):
        [chunk async for chunk in runnable.astream(1)]

    assert _lambda_mock.call_count == 1
    _lambda_mock.reset_mock()
    attempt = 0

    chunks = [
        c
        async for c in runnable.with_retry(
            stop_after_attempt=2,
            retry_if_exception_type=(ValueError,),
        ).astream(3)
    ]

    assert chunks == [1, RunnableStreamResetMarker(), 1, 3]
    assert _lambda_mock.call_count == 2  # retried
    _lambda_mock.reset_mock()
    attempt = 0
