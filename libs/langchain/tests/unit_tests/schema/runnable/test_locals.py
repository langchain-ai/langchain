from typing import Any, Callable, Type

import pytest

from langchain import PromptTemplate
from langchain.llms import FakeListLLM
from langchain.schema.runnable import (
    GetLocalVar,
    PutLocalVar,
    RunnablePassthrough,
    RunnableSequence,
)


@pytest.mark.parametrize(
    ("method", "input", "output"),
    [
        (lambda r, x: r.invoke(x), "foo", "foo"),
        (lambda r, x: r.batch(x), ["foo", "bar"], ["foo", "bar"]),
        (lambda r, x: list(r.stream(x))[0], "foo", "foo"),
    ],
)
def test_put_get(method: Callable, input: Any, output: Any) -> None:
    runnable = PutLocalVar("input") | GetLocalVar("input")
    assert method(runnable, input) == output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "input", "output"),
    [
        (lambda r, x: r.ainvoke(x), "foo", "foo"),
        (lambda r, x: r.abatch(x), ["foo", "bar"], ["foo", "bar"]),
    ],
)
async def test_put_get_async(method: Callable, input: Any, output: Any) -> None:
    runnable = PutLocalVar("input") | GetLocalVar("input")
    assert await method(runnable, input) == output


@pytest.mark.parametrize(
    ("runnable", "error"),
    [
        (PutLocalVar("input"), ValueError),
        (GetLocalVar("input"), ValueError),
        (PutLocalVar("input") | GetLocalVar("missing"), KeyError),
    ],
)
def test_incorrect_usage(runnable: RunnableSequence, error: Type[Exception]) -> None:
    with pytest.raises(error):
        runnable.invoke("foo")


def test_get_in_map() -> None:
    runnable: RunnableSequence = PutLocalVar("input") | {"bar": GetLocalVar("input")}
    assert runnable.invoke("foo") == {"bar": "foo"}


def test_put_in_map() -> None:
    runnable: RunnableSequence = {"bar": PutLocalVar("input")} | GetLocalVar("input")
    with pytest.raises(KeyError):
        runnable.invoke("foo")


@pytest.mark.parametrize(
    "runnable",
    [
        PutLocalVar("input") | GetLocalVar("input", passthrough_key="output"),
        (
            PutLocalVar("input")
            | {"input": RunnablePassthrough()}
            | PromptTemplate.from_template("say {input}")
            | FakeListLLM(responses=["hello"])
            | GetLocalVar("input", passthrough_key="output")
        ),
    ],
)
@pytest.mark.parametrize(
    ("method", "input", "output"),
    [
        (lambda r, x: r.invoke(x), "hello", {"input": "hello", "output": "hello"}),
        (lambda r, x: r.batch(x), ["hello"], [{"input": "hello", "output": "hello"}]),
        (
            lambda r, x: list(r.stream(x))[0],
            "hello",
            {"input": "hello", "output": "hello"},
        ),
    ],
)
def test_put_get_sequence(
    runnable: RunnableSequence, method: Callable, input: Any, output: Any
) -> None:
    assert method(runnable, input) == output
