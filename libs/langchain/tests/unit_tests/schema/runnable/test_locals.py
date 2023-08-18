import pytest

from langchain import PromptTemplate
from langchain.llms import FakeListLLM
from langchain.schema.runnable import (
    GetLocalVar,
    PutLocalVar,
    RunnablePassthrough,
    RunnableSequence,
)


@pytest.mark.asyncio
async def test_put_get() -> None:
    runnable = PutLocalVar("input") | GetLocalVar("input")
    assert runnable.invoke("foo") == "foo"
    assert runnable.batch(["foo", "bar"]) == ["foo", "bar"]
    assert list(runnable.stream("foo"))[0] == "foo"

    assert await runnable.ainvoke("foo") == "foo"
    assert await runnable.abatch(["foo", "bar"]) == ["foo", "bar"]
    async for x in runnable.astream("foo"):
        assert x == "foo"


def test_missing_config() -> None:
    with pytest.raises(ValueError):
        PutLocalVar("input").invoke("foo")

    with pytest.raises(ValueError):
        GetLocalVar[str, str]("input").invoke("foo")


def test_get_missing_var_invoke() -> None:
    runnable = PutLocalVar("input") | GetLocalVar("missing")
    with pytest.raises(KeyError):
        runnable.invoke("foo")


def test_get_in_map() -> None:
    runnable: RunnableSequence = PutLocalVar("input") | {"bar": GetLocalVar("input")}
    assert runnable.invoke("foo") == {"bar": "foo"}


def test_cant_put_in_map() -> None:
    runnable: RunnableSequence = {"bar": PutLocalVar("input")} | GetLocalVar("input")
    with pytest.raises(KeyError):
        runnable.invoke("foo")


def test_get_passthrough_key() -> None:
    runnable = PutLocalVar("input") | GetLocalVar("input", passthrough_key="output")
    assert runnable.invoke("foo") == {"input": "foo", "output": "foo"}


def test_multi_step_sequence() -> None:
    prompt = PromptTemplate.from_template("say {foo}")
    runnable = (
        PutLocalVar("foo")
        | {"foo": RunnablePassthrough()}
        | prompt
        | FakeListLLM(responses=["bar"])
        | GetLocalVar("foo", passthrough_key="output")
    )
    assert runnable.invoke("hello") == {"foo": "hello", "output": "bar"}
