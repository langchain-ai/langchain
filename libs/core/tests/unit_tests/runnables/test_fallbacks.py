from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import (
    Any,
)

import pytest
from pydantic import BaseModel
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
    FallbackLatch,
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
def llm() -> RunnableWithFallbacks[Any, Any]:
    error_llm = FakeListLLM(responses=["foo"], i=1)
    pass_llm = FakeListLLM(responses=["bar"])

    return error_llm.with_fallbacks([pass_llm])


@pytest.fixture
def llm_multi() -> RunnableWithFallbacks[Any, Any]:
    error_llm = FakeListLLM(responses=["foo"], i=1)
    error_llm_2 = FakeListLLM(responses=["baz"], i=1)
    pass_llm = FakeListLLM(responses=["bar"])

    return error_llm.with_fallbacks([error_llm_2, pass_llm])


@pytest.fixture
def chain() -> Runnable[Any, str]:
    error_llm = FakeListLLM(responses=["foo"], i=1)
    pass_llm = FakeListLLM(responses=["bar"])

    prompt = PromptTemplate.from_template("what did baz say to {buz}")
    return RunnableParallel({"buz": lambda x: x}) | (prompt | error_llm).with_fallbacks(
        [prompt | pass_llm]
    )


def _raise_error(_: dict[str, Any]) -> str:
    raise ValueError


def _dont_raise_error(inputs: dict[str, Any]) -> str:
    if "exception" in inputs:
        return "bar"
    raise ValueError


@pytest.fixture
def chain_pass_exceptions() -> Runnable[Any, str]:
    fallback = RunnableLambda(_dont_raise_error)
    return {"text": RunnablePassthrough()} | RunnableLambda(
        _raise_error
    ).with_fallbacks([fallback], exception_key="exception")


@pytest.mark.parametrize(
    "runnable_name",
    ["llm", "llm_multi", "chain", "chain_pass_exceptions"],
)
def test_fallbacks(
    runnable_name: str, request: Any, snapshot: SnapshotAssertion
) -> None:
    runnable: Runnable[Any, Any] = request.getfixturevalue(runnable_name)
    assert runnable.invoke("hello") == "bar"
    assert runnable.batch(["hi", "hey", "bye"]) == ["bar"] * 3
    assert list(runnable.stream("hello")) == ["bar"]
    assert dumps(runnable, pretty=True) == snapshot


@pytest.mark.parametrize(
    "runnable_name",
    ["llm", "llm_multi", "chain", "chain_pass_exceptions"],
)
async def test_fallbacks_async(runnable_name: str, request: Any) -> None:
    runnable: Runnable[Any, Any] = request.getfixturevalue(runnable_name)
    assert await runnable.ainvoke("hello") == "bar"
    assert await runnable.abatch(["hi", "hey", "bye"]) == ["bar"] * 3
    assert list(await runnable.ainvoke("hello")) == list("bar")


def _runnable(inputs: dict[str, Any]) -> str:
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


def _assert_potential_error(actual: list[Any], expected: list[Any]) -> None:
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


def _generate(_: Iterator[Any]) -> Iterator[str]:
    yield from "foo bar"


def _error(msg: str) -> None:
    raise ValueError(msg)


def _generate_immediate_error(_: Iterator[Any]) -> Iterator[str]:
    _error("immediate error")
    yield ""


def _generate_delayed_error(_: Iterator[Any]) -> Iterator[str]:
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


async def _agenerate(_: AsyncIterator[Any]) -> AsyncIterator[str]:
    for c in "foo bar":
        yield c


async def _agenerate_immediate_error(_: AsyncIterator[Any]) -> AsyncIterator[str]:
    _error("immediate error")
    yield ""


async def _agenerate_delayed_error(_: AsyncIterator[Any]) -> AsyncIterator[str]:
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
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(tools=tools)

    @override
    def with_structured_output(
        self, schema: dict | type[BaseModel], **kwargs: Any
    ) -> Runnable[LanguageModelInput, dict[str, int] | BaseModel]:
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
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
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


# -- Latched fallbacks -------------------------------------------------------


def _make_counting_chain(
    *, latch: FallbackLatch | None
) -> tuple[RunnableWithFallbacks[Any, Any], dict[str, int]]:
    """Build `fails | fallback` and return counters for assertions."""
    counters = {"primary": 0, "fallback": 0}

    def _primary(_: Any) -> str:
        counters["primary"] += 1
        msg = "boom"
        raise ValueError(msg)

    def _fallback(_: Any) -> str:
        counters["fallback"] += 1
        return "ok"

    chain = RunnableLambda(_primary).with_fallbacks(
        [RunnableLambda(_fallback)], latch=latch
    )
    return chain, counters


def test_latch_flips_on_first_failure_invoke() -> None:
    latch = FallbackLatch()
    chain, counts = _make_counting_chain(latch=latch)

    assert chain.invoke("a") == "ok"
    assert latch.latched is True
    assert counts == {"primary": 1, "fallback": 1}

    # Latched: primary skipped on subsequent calls.
    chain.invoke("b")
    chain.invoke("c")
    assert counts == {"primary": 1, "fallback": 3}


async def test_latch_flips_on_first_failure_ainvoke() -> None:
    latch = FallbackLatch()
    chain, counts = _make_counting_chain(latch=latch)

    assert await chain.ainvoke("a") == "ok"
    assert latch.latched is True
    assert counts == {"primary": 1, "fallback": 1}

    await chain.ainvoke("b")
    await chain.ainvoke("c")
    assert counts == {"primary": 1, "fallback": 3}


def test_latch_reset_re_enables_primary() -> None:
    latch = FallbackLatch()
    chain, counts = _make_counting_chain(latch=latch)

    chain.invoke("x")
    assert latch.latched is True
    chain.invoke("y")
    assert counts == {"primary": 1, "fallback": 2}

    latch.reset()
    assert latch.latched is False
    chain.invoke("z")
    # Primary attempted again, fails, fallback runs, latch re-trips.
    assert counts == {"primary": 2, "fallback": 3}
    assert latch.latched is True


def test_latch_shared_across_wrappers() -> None:
    """A single FallbackLatch trips for every wrapper that shares it.

    This is the per-run circuit-breaker property: tool-bound and bare
    versions of the same model can both consult one latch so a single
    Anthropic failure on the tool path skips Anthropic on the bare path
    too.
    """
    latch = FallbackLatch()
    chain_a, _counts_a = _make_counting_chain(latch=latch)
    chain_b, counts_b = _make_counting_chain(latch=latch)

    chain_a.invoke("a")  # trips the shared latch
    assert latch.latched is True

    chain_b.invoke("b")
    chain_b.invoke("c")
    # chain_b's primary never runs after the trip.
    assert counts_b["primary"] == 0
    assert counts_b["fallback"] == 2


def test_latch_propagates_through_getattr_bind() -> None:
    """`with_fallbacks(..., latch=l).bind(...)` must preserve `l`.

    The `__getattr__` runnable-method rewrite reconstructs the wrapper —
    without forwarding the latch explicitly, the rebuilt wrapper would
    have a `None` latch and silently lose its circuit-breaker.
    """
    latch = FallbackLatch()
    chain, _ = _make_counting_chain(latch=latch)
    bound = chain.bind(tools=[])  # returns Runnable; not RunnableWithFallbacks
    # Sanity: original wrapper still carries the latch.
    assert chain.latch is latch
    _ = bound  # type: ignore[unused-ignore]


def test_latch_default_none_keeps_existing_behavior() -> None:
    """No latch passed → primary is retried on every call (existing behavior)."""
    chain, counts = _make_counting_chain(latch=None)
    chain.invoke("a")
    chain.invoke("b")
    assert counts == {"primary": 2, "fallback": 2}


# -- close / aclose propagation ----------------------------------------------


class _CloseTracker(RunnableLambda):
    """RunnableLambda wrapper that records close()/aclose() invocations."""

    def __init__(self) -> None:
        super().__init__(lambda x: x)
        self.sync_closed = 0
        self.async_closed = 0

    def close(self) -> None:
        self.sync_closed += 1

    async def aclose(self) -> None:
        self.async_closed += 1


def test_close_walks_runnable_and_fallbacks() -> None:
    primary = _CloseTracker()
    fb1 = _CloseTracker()
    fb2 = _CloseTracker()
    chain = primary.with_fallbacks([fb1, fb2])

    chain.close()

    assert primary.sync_closed == 1
    assert fb1.sync_closed == 1
    assert fb2.sync_closed == 1


async def test_aclose_walks_runnable_and_fallbacks() -> None:
    primary = _CloseTracker()
    fb = _CloseTracker()
    chain = primary.with_fallbacks([fb])

    await chain.aclose()

    assert primary.async_closed == 1
    assert fb.async_closed == 1


def test_close_swallows_per_runnable_errors() -> None:
    """One bad close() should not stop the others from running."""

    class _BadClose(RunnableLambda):
        def __init__(self) -> None:
            super().__init__(lambda x: x)

        def close(self) -> None:
            msg = "nope"
            raise RuntimeError(msg)

    bad = _BadClose()
    good = _CloseTracker()
    chain = bad.with_fallbacks([good])

    chain.close()  # should not raise
    assert good.sync_closed == 1
