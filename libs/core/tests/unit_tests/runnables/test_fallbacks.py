import sys
from typing import Any, AsyncIterator, Iterator

import pytest
from syrupy import SnapshotAssertion

from langchain_core.language_models import FakeListLLM
from langchain_core.load import dumps
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableGenerator,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableWithFallbacks,
)


@pytest.fixture()
def llm() -> RunnableWithFallbacks:
    error_llm = FakeListLLM(responses=["foo"], i=1)
    pass_llm = FakeListLLM(responses=["bar"])

    return error_llm.with_fallbacks([pass_llm])


@pytest.fixture()
def llm_multi() -> RunnableWithFallbacks:
    error_llm = FakeListLLM(responses=["foo"], i=1)
    error_llm_2 = FakeListLLM(responses=["baz"], i=1)
    pass_llm = FakeListLLM(responses=["bar"])

    return error_llm.with_fallbacks([error_llm_2, pass_llm])


@pytest.fixture()
def chain() -> Runnable:
    error_llm = FakeListLLM(responses=["foo"], i=1)
    pass_llm = FakeListLLM(responses=["bar"])

    prompt = PromptTemplate.from_template("what did baz say to {buz}")
    return RunnableParallel({"buz": lambda x: x}) | (prompt | error_llm).with_fallbacks(
        [prompt | pass_llm]
    )


def _raise_error(inputs: dict) -> str:
    raise ValueError()


def _dont_raise_error(inputs: dict) -> str:
    if "exception" in inputs:
        return "bar"
    raise ValueError()


@pytest.fixture()
def chain_pass_exceptions() -> Runnable:
    fallback = RunnableLambda(_dont_raise_error)
    return {"text": RunnablePassthrough()} | RunnableLambda(
        _raise_error
    ).with_fallbacks([fallback], exception_key="exception")


@pytest.mark.parametrize(
    "runnable",
    ["llm", "llm_multi", "chain", "chain_pass_exceptions"],
)
async def test_fallbacks(
    runnable: RunnableWithFallbacks, request: Any, snapshot: SnapshotAssertion
) -> None:
    runnable = request.getfixturevalue(runnable)
    assert runnable.invoke("hello") == "bar"
    assert runnable.batch(["hi", "hey", "bye"]) == ["bar"] * 3
    assert list(runnable.stream("hello")) == ["bar"]
    assert await runnable.ainvoke("hello") == "bar"
    assert await runnable.abatch(["hi", "hey", "bye"]) == ["bar"] * 3
    assert list(await runnable.ainvoke("hello")) == list("bar")
    if sys.version_info >= (3, 9):
        assert dumps(runnable, pretty=True) == snapshot


def _runnable(inputs: dict) -> str:
    if inputs["text"] == "foo":
        return "first"
    if "exception" not in inputs:
        raise ValueError()
    if inputs["text"] == "bar":
        return "second"
    if isinstance(inputs["exception"], ValueError):
        raise RuntimeError()
    return "third"


def _assert_potential_error(actual: list, expected: list) -> None:
    for x, y in zip(actual, expected):
        if isinstance(x, Exception):
            assert isinstance(y, type(x))
        else:
            assert x == y


def test_invoke_with_exception_key() -> None:
    runnable = RunnableLambda(_runnable)
    runnable_with_single = runnable.with_fallbacks(
        [runnable], exception_key="exception"
    )
    with pytest.raises(ValueError):
        runnable_with_single.invoke({"text": "baz"})

    actual = runnable_with_single.invoke({"text": "bar"})
    expected = "second"
    _assert_potential_error([actual], [expected])

    runnable_with_double = runnable.with_fallbacks(
        [runnable, runnable], exception_key="exception"
    )
    actual = runnable_with_double.invoke({"text": "baz"})

    expected = "third"
    _assert_potential_error([actual], [expected])


async def test_ainvoke_with_exception_key() -> None:
    runnable = RunnableLambda(_runnable)
    runnable_with_single = runnable.with_fallbacks(
        [runnable], exception_key="exception"
    )
    with pytest.raises(ValueError):
        await runnable_with_single.ainvoke({"text": "baz"})

    actual = await runnable_with_single.ainvoke({"text": "bar"})
    expected = "second"
    _assert_potential_error([actual], [expected])

    runnable_with_double = runnable.with_fallbacks(
        [runnable, runnable], exception_key="exception"
    )
    actual = await runnable_with_double.ainvoke({"text": "baz"})
    expected = "third"
    _assert_potential_error([actual], [expected])


def test_batch() -> None:
    runnable = RunnableLambda(_runnable)
    with pytest.raises(ValueError):
        runnable.batch([{"text": "foo"}, {"text": "bar"}, {"text": "baz"}])
    actual = runnable.batch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )
    expected = ["first", ValueError(), ValueError()]
    _assert_potential_error(actual, expected)

    runnable_with_single = runnable.with_fallbacks(
        [runnable], exception_key="exception"
    )
    with pytest.raises(RuntimeError):
        runnable_with_single.batch([{"text": "foo"}, {"text": "bar"}, {"text": "baz"}])
    actual = runnable_with_single.batch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )
    expected = ["first", "second", RuntimeError()]
    _assert_potential_error(actual, expected)

    runnable_with_double = runnable.with_fallbacks(
        [runnable, runnable], exception_key="exception"
    )
    actual = runnable_with_double.batch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )

    expected = ["first", "second", "third"]
    _assert_potential_error(actual, expected)

    runnable_with_double = runnable.with_fallbacks(
        [runnable, runnable],
        exception_key="exception",
        exceptions_to_handle=(ValueError,),
    )
    actual = runnable_with_double.batch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )

    expected = ["first", "second", RuntimeError()]
    _assert_potential_error(actual, expected)


async def test_abatch() -> None:
    runnable = RunnableLambda(_runnable)
    with pytest.raises(ValueError):
        await runnable.abatch([{"text": "foo"}, {"text": "bar"}, {"text": "baz"}])
    actual = await runnable.abatch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )
    expected = ["first", ValueError(), ValueError()]
    _assert_potential_error(actual, expected)

    runnable_with_single = runnable.with_fallbacks(
        [runnable], exception_key="exception"
    )
    with pytest.raises(RuntimeError):
        await runnable_with_single.abatch(
            [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}]
        )
    actual = await runnable_with_single.abatch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )
    expected = ["first", "second", RuntimeError()]
    _assert_potential_error(actual, expected)

    runnable_with_double = runnable.with_fallbacks(
        [runnable, runnable], exception_key="exception"
    )
    actual = await runnable_with_double.abatch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )

    expected = ["first", "second", "third"]
    _assert_potential_error(actual, expected)

    runnable_with_double = runnable.with_fallbacks(
        [runnable, runnable],
        exception_key="exception",
        exceptions_to_handle=(ValueError,),
    )
    actual = await runnable_with_double.abatch(
        [{"text": "foo"}, {"text": "bar"}, {"text": "baz"}], return_exceptions=True
    )

    expected = ["first", "second", RuntimeError()]
    _assert_potential_error(actual, expected)


def _generate(input: Iterator) -> Iterator[str]:
    yield from "foo bar"


def _generate_immediate_error(input: Iterator) -> Iterator[str]:
    raise ValueError()
    yield ""


def _generate_delayed_error(input: Iterator) -> Iterator[str]:
    yield ""
    raise ValueError()


def test_fallbacks_stream() -> None:
    runnable = RunnableGenerator(_generate_immediate_error).with_fallbacks(
        [RunnableGenerator(_generate)]
    )
    assert list(runnable.stream({})) == [c for c in "foo bar"]

    with pytest.raises(ValueError):
        runnable = RunnableGenerator(_generate_delayed_error).with_fallbacks(
            [RunnableGenerator(_generate)]
        )
        list(runnable.stream({}))


async def _agenerate(input: AsyncIterator) -> AsyncIterator[str]:
    for c in "foo bar":
        yield c


async def _agenerate_immediate_error(input: AsyncIterator) -> AsyncIterator[str]:
    raise ValueError()
    yield ""


async def _agenerate_delayed_error(input: AsyncIterator) -> AsyncIterator[str]:
    yield ""
    raise ValueError()


async def test_fallbacks_astream() -> None:
    runnable = RunnableGenerator(_agenerate_immediate_error).with_fallbacks(
        [RunnableGenerator(_agenerate)]
    )
    expected = (c for c in "foo bar")
    async for c in runnable.astream({}):
        assert c == next(expected)

    with pytest.raises(ValueError):
        runnable = RunnableGenerator(_agenerate_delayed_error).with_fallbacks(
            [RunnableGenerator(_agenerate)]
        )
        async for c in runnable.astream({}):
            pass
