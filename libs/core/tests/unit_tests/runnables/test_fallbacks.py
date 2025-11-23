from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from typing import (
    Any,
)

import pytest
from syrupy.assertion import SnapshotAssertion
from typing_extensions import override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import (
    BaseChatModel,
    FakeListLLM,
    LanguageModelInput,
)
from langchain_core.load import dumps
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableGenerator,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableWithFallbacks,
)
from langchain_core.tools import BaseTool


@pytest.fixture
def llm() -> RunnableWithFallbacks:
    error_llm = FakeListLLM(responses=["foo"], i=1)
    pass_llm = FakeListLLM(responses=["bar"])

    return error_llm.with_fallbacks([pass_llm])


@pytest.fixture
def llm_multi() -> RunnableWithFallbacks:
    error_llm = FakeListLLM(responses=["foo"], i=1)
    error_llm_2 = FakeListLLM(responses=["baz"], i=1)
    pass_llm = FakeListLLM(responses=["bar"])

    return error_llm.with_fallbacks([error_llm_2, pass_llm])


@pytest.fixture
def chain() -> Runnable:
    error_llm = FakeListLLM(responses=["foo"], i=1)
    pass_llm = FakeListLLM(responses=["bar"])

    prompt = PromptTemplate.from_template("what did baz say to {buz}")
    return RunnableParallel({"buz": lambda x: x}) | (prompt | error_llm).with_fallbacks(
        [prompt | pass_llm]
    )


def _raise_error(_: dict) -> str:
    raise ValueError


def _dont_raise_error(inputs: dict) -> str:
    if "exception" in inputs:
        return "bar"
    raise ValueError


@pytest.fixture
def chain_pass_exceptions() -> Runnable:
    fallback = RunnableLambda(_dont_raise_error)
    return {"text": RunnablePassthrough()} | RunnableLambda(
        _raise_error
    ).with_fallbacks([fallback], exception_key="exception")


@pytest.mark.parametrize(
    "runnable",
    ["llm", "llm_multi", "chain", "chain_pass_exceptions"],
)
def test_fallbacks(
    runnable: RunnableWithFallbacks, request: Any, snapshot: SnapshotAssertion
) -> None:
    runnable = request.getfixturevalue(runnable)
    assert runnable.invoke("hello") == "bar"
    assert runnable.batch(["hi", "hey", "bye"]) == ["bar"] * 3
    assert list(runnable.stream("hello")) == ["bar"]
    assert dumps(runnable, pretty=True) == snapshot


@pytest.mark.parametrize(
    "runnable",
    ["llm", "llm_multi", "chain", "chain_pass_exceptions"],
)
async def test_fallbacks_async(runnable: RunnableWithFallbacks, request: Any) -> None:
    runnable = request.getfixturevalue(runnable)
    assert await runnable.ainvoke("hello") == "bar"
    assert await runnable.abatch(["hi", "hey", "bye"]) == ["bar"] * 3
    assert list(await runnable.ainvoke("hello")) == list("bar")


def _runnable(inputs: dict) -> str:
    if inputs["text"] == "foo":
        return "first"
    if "exception" not in inputs:
        msg = "missing exception"
        raise ValueError(msg)
    if inputs["text"] == "bar":
        return "second"
    if isinstance(inputs["exception"], ValueError):
        raise RuntimeError  # noqa: TRY004
    return "third"


def _assert_potential_error(actual: list, expected: list) -> None:
    for x, y in zip(actual, expected, strict=False):
        if isinstance(x, Exception):
            assert isinstance(y, type(x))
        else:
            assert x == y


def test_invoke_with_exception_key() -> None:
    runnable = RunnableLambda(_runnable)
    runnable_with_single = runnable.with_fallbacks(
        [runnable], exception_key="exception"
    )
    with pytest.raises(ValueError, match="missing exception"):
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
    with pytest.raises(ValueError, match="missing exception"):
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
    with pytest.raises(ValueError, match="missing exception"):
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
    with pytest.raises(ValueError, match="missing exception"):
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
            [
                {"text": "foo"},
                {"text": "bar"},
                {"text": "baz"},
            ]
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


def _generate(_: Iterator) -> Iterator[str]:
    yield from "foo bar"


def _error(msg: str) -> None:
    raise ValueError(msg)


def _generate_immediate_error(_: Iterator) -> Iterator[str]:
    _error("immediate error")
    yield ""


def _generate_delayed_error(_: Iterator) -> Iterator[str]:
    yield ""
    _error("delayed error")


def test_fallbacks_stream() -> None:
    runnable = RunnableGenerator(_generate_immediate_error).with_fallbacks(
        [RunnableGenerator(_generate)]
    )
    assert list(runnable.stream({})) == list("foo bar")

    runnable = RunnableGenerator(_generate_delayed_error).with_fallbacks(
        [RunnableGenerator(_generate)]
    )
    with pytest.raises(ValueError, match="delayed error"):
        list(runnable.stream({}))


async def _agenerate(_: AsyncIterator) -> AsyncIterator[str]:
    for c in "foo bar":
        yield c


async def _agenerate_immediate_error(_: AsyncIterator) -> AsyncIterator[str]:
    _error("immediate error")
    yield ""


async def _agenerate_delayed_error(_: AsyncIterator) -> AsyncIterator[str]:
    yield ""
    _error("delayed error")


async def test_fallbacks_astream() -> None:
    runnable = RunnableGenerator(_agenerate_immediate_error).with_fallbacks(
        [RunnableGenerator(_agenerate)]
    )
    expected = (c for c in "foo bar")
    async for c in runnable.astream({}):
        assert c == next(expected)

    runnable = RunnableGenerator(_agenerate_delayed_error).with_fallbacks(
        [RunnableGenerator(_agenerate)]
    )
    with pytest.raises(ValueError, match="delayed error"):
        _ = [_ async for _ in runnable.astream({})]


class FakeStructuredOutputModel(BaseChatModel):
    foo: int

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call."""
        return ChatResult(generations=[])

    @override
    def bind_tools(
        self,
        tools: Sequence[Mapping[str, Any] | type | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(tools=tools)

    @override
    def with_structured_output(
        self, schema: Mapping | type, **kwargs: Any
    ) -> Runnable[LanguageModelInput, Any]:
        return RunnableLambda(lambda _: {"foo": self.foo})

    @property
    def _llm_type(self) -> str:
        return "fake1"


class FakeModel(BaseChatModel):
    bar: int

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call."""
        return ChatResult(generations=[])

    @override
    def bind_tools(
        self,
        tools: Sequence[Mapping[str, Any] | type | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(tools=tools)

    @property
    def _llm_type(self) -> str:
        return "fake2"


def test_fallbacks_getattr() -> None:
    llm_with_fallbacks = FakeStructuredOutputModel(foo=3).with_fallbacks(
        [FakeModel(bar=4)]
    )
    assert llm_with_fallbacks.foo == 3

    with pytest.raises(AttributeError):
        assert llm_with_fallbacks.bar == 4


def test_fallbacks_getattr_runnable_output() -> None:
    llm_with_fallbacks = FakeStructuredOutputModel(foo=3).with_fallbacks(
        [FakeModel(bar=4)]
    )
    llm_with_fallbacks_with_tools = llm_with_fallbacks.bind_tools([])
    assert isinstance(llm_with_fallbacks_with_tools, RunnableWithFallbacks)
    assert isinstance(llm_with_fallbacks_with_tools.runnable, RunnableBinding)
    assert all(
        isinstance(fallback, RunnableBinding)
        for fallback in llm_with_fallbacks_with_tools.fallbacks
    )
    assert llm_with_fallbacks_with_tools.runnable.kwargs["tools"] == []
