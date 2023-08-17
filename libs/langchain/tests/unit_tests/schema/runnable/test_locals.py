import pytest

from langchain.schema.runnable import GetLocalVar, PutLocalVar


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
        GetLocalVar("input").invoke("foo")


def test_get_missing_var_invoke() -> None:
    runnable = PutLocalVar("input") | GetLocalVar("missing")
    with pytest.raises(KeyError):
        runnable.invoke("foo")

