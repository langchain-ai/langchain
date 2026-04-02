from typing import Any

import pytest

from langchain_core.exceptions import OutputVerificationError
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableWithOutputVerification,
)


def test_invoke_verified() -> None:
    inner: Runnable[int, int] = RunnableLambda(lambda x: x * 2)
    gated = inner.with_output_verification(verify=lambda y: y == 4)
    assert gated.invoke(2) == 4


def test_invoke_blocked() -> None:
    inner: Runnable[Any, str] = RunnableLambda(lambda _: "Fatal Error: nope")
    gated = inner.with_output_verification(
        verify=lambda y: "error" not in str(y).lower(),
        step_name="tool_a",
    )
    with pytest.raises(OutputVerificationError) as exc_info:
        gated.invoke({})
    assert exc_info.value.step_name == "tool_a"
    assert "error" in str(exc_info.value.output).lower()


def test_audit_sink_and_on_audit() -> None:
    sink: list[dict] = []
    side: list[dict] = []

    def _on_audit(entry: dict) -> None:
        side.append(entry)

    inner = RunnableLambda(lambda x: x)
    gated = inner.with_output_verification(
        verify=lambda y: y > 0,
        step_name="step-1",
        audit_sink=sink,
        on_audit=_on_audit,
    )
    assert gated.invoke(3) == 3
    assert sink[0]["status"] == "VERIFIED"
    assert sink[0]["step"] == "step-1"
    assert side[0]["status"] == "VERIFIED"

    with pytest.raises(OutputVerificationError):
        gated.invoke(-1)
    assert sink[-1]["status"] == "BLOCKED"
    assert side[-1]["status"] == "BLOCKED"


def test_batch() -> None:
    inner = RunnableLambda(lambda x: x)
    gated = inner.with_output_verification(verify=lambda y: y % 2 == 0)
    assert gated.batch([2, 4]) == [2, 4]
    with pytest.raises(OutputVerificationError):
        gated.batch([2, 3])


def test_stream_aggregates_then_verifies() -> None:
    model = GenericFakeChatModel(
        messages=iter(["hello world"]),
    )
    gated = model.with_output_verification(
        verify=lambda m: "error" not in str(getattr(m, "content", m)).lower(),
        step_name="chat",
    )
    chunks = list(gated.stream("hi"))
    assert len(chunks) >= 1
    assert all("error" not in str(c.content).lower() for c in chunks)


def test_stream_blocked_before_yield() -> None:
    model = GenericFakeChatModel(
        messages=iter(["error: boom"]),
    )
    gated = model.with_output_verification(
        verify=lambda m: "error" not in str(getattr(m, "content", m)).lower(),
    )
    with pytest.raises(OutputVerificationError):
        list(gated.stream("hi"))


@pytest.mark.asyncio
async def test_astream_verified() -> None:
    model = GenericFakeChatModel(
        messages=iter(["ok"]),
    )
    gated = model.with_output_verification(
        verify=lambda m: "error" not in str(getattr(m, "content", m)).lower(),
    )
    chunks = [c async for c in gated.astream("hi")]
    assert len(chunks) >= 1


def test_runnable_binding_forwards_verification() -> None:
    inner = RunnableLambda(lambda x: x).with_config(run_name="inner")
    gated = inner.with_output_verification(verify=lambda y: y > 0)
    assert gated.invoke(5) == 5


def test_direct_constructor_matches_factory() -> None:
    inner: Runnable[int, int] = RunnableLambda(lambda x: x)
    direct: RunnableWithOutputVerification[int, int] = RunnableWithOutputVerification(
        bound=inner,
        verify=lambda y: y == 7,
        kwargs={},
        config={},
    )
    assert direct.invoke(7) == 7
