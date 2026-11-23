"""Tests for the v3 dispatch path on Runnable.astream_events / stream_events."""

from __future__ import annotations

import warnings
from typing import Literal, get_overloads

import pytest

from langchain_core._api import LangChainDeprecationWarning
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("version", [None, "v2"])
async def test_v2_chat_model_events_are_not_deprecated(
    version: Literal["v2"] | None, *, bound: bool
) -> None:
    base_model = FakeListChatModel(responses=["hello"])
    model = base_model.bind() if bound else base_model
    v2_overload = get_overloads(type(model).astream_events)[0]
    assert "v2" in v2_overload.__annotations__["version"]
    assert not hasattr(v2_overload, "__deprecated__")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stream = (
            model.astream_events("hi")
            if version is None
            else model.astream_events("hi", version=version)
        )
        events = [event async for event in stream]
    assert [event["event"] for event in events] == [
        "on_chat_model_start",
        *["on_chat_model_stream"] * len("hello"),
        "on_chat_model_end",
    ]
    assert not any(
        issubclass(warning.category, LangChainDeprecationWarning) for warning in caught
    )


def _double(x: int) -> int:
    return x * 2


def test_v3_on_plain_runnable_raises_not_implemented_sync() -> None:
    runnable = RunnableLambda(_double)
    with pytest.raises(NotImplementedError, match="v3"):
        runnable.stream_events(2, version="v3")


async def test_v3_on_plain_runnable_raises_not_implemented_async() -> None:
    runnable = RunnableLambda(_double)
    with pytest.raises(NotImplementedError, match="v3"):
        await runnable.astream_events(2, version="v3")
