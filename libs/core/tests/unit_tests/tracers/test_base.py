"""Parity tests for `BaseTracer` and `AsyncBaseTracer`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import pytest
from typing_extensions import override

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.load.dump import dumpd
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult
from langchain_core.tools import tool
from langchain_core.tracers.base import AsyncBaseTracer, BaseTracer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.messages import BaseMessage
    from langchain_core.tracers.schemas import Run


@tool
def _add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class _SyncTracer(BaseTracer):
    def __init__(self) -> None:
        super().__init__()
        self.runs: list[Run] = []
        self.error_outputs: list[dict[str, Any] | None] = []

    @override
    def _persist_run(self, run: Run) -> None:
        self.runs.append(run)

    @override
    def _on_llm_error(self, run: Run) -> None:
        self.error_outputs.append(run.outputs)


class _AsyncTracer(AsyncBaseTracer):
    def __init__(self) -> None:
        super().__init__()
        self.runs: list[Run] = []
        self.error_outputs: list[dict[str, Any] | None] = []

    @override
    async def _persist_run(self, run: Run) -> None:
        self.runs.append(run)

    @override
    async def _on_llm_error(self, run: Run) -> None:
        self.error_outputs.append(run.outputs)


class _FailsMidStream(GenericFakeChatModel):
    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="partial"))
        msg = "boom"
        raise ValueError(msg)


def _partial_content(outputs: dict[str, Any]) -> str:
    generation = outputs["generations"][0][0]
    return cast("str", generation["message"]["kwargs"]["content"])


async def test_sync_and_async_tracers_respect_configured_tool_name() -> None:
    names: list[str] = []
    _add.with_listeners(on_end=lambda run: names.append(run.name)).invoke(
        {"a": 1, "b": 2},
        config={"run_name": "renamed"},
    )

    async def _on_end(run: Run) -> None:
        names.append(run.name)

    await _add.with_alisteners(on_end=_on_end).ainvoke(
        {"a": 1, "b": 2},
        config={"run_name": "renamed"},
    )
    assert names == ["renamed", "renamed"]

    run_id = uuid4()
    sync_tracer = _SyncTracer()
    sync_tracer.on_tool_start(
        serialized={"name": "add"},
        input_str="test",
        run_id=run_id,
        name="renamed",
    )
    sync_tracer.on_tool_end("ok", run_id=run_id)

    async_run_id = uuid4()
    async_tracer = _AsyncTracer()
    await async_tracer.on_tool_start(
        serialized={"name": "add"},
        input_str="test",
        run_id=async_run_id,
        name="renamed",
    )
    await async_tracer.on_tool_end("ok", run_id=async_run_id)
    assert sync_tracer.runs[0].name == async_tracer.runs[0].name == "renamed"


async def test_sync_and_async_tracers_keep_partial_llm_stream_on_error() -> None:
    sync_tracer = _SyncTracer()
    async_tracer = _AsyncTracer()
    model = _FailsMidStream(messages=iter([]))

    with pytest.raises(ValueError, match="boom"):
        list(model.stream("hi", config={"callbacks": [sync_tracer]}))
    with pytest.raises(ValueError, match="boom"):
        async for _ in model.astream("hi", config={"callbacks": [async_tracer]}):
            pass

    assert sync_tracer.error_outputs[0] is not None
    assert async_tracer.error_outputs[0] is not None
    assert "generations" in sync_tracer.error_outputs[0]
    assert "generations" in async_tracer.error_outputs[0]
    assert _partial_content(sync_tracer.error_outputs[0]) == "partial"
    assert _partial_content(async_tracer.error_outputs[0]) == "partial"

    response = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="partial"))]]
    )
    exception = ValueError("boom")
    sync_run_id = uuid4()
    async_run_id = uuid4()
    sync_direct = _SyncTracer()
    async_direct = _AsyncTracer()
    sync_direct.on_llm_start(serialized={"id": ["llm"]}, prompts=[], run_id=sync_run_id)
    sync_direct.on_llm_error(exception, run_id=sync_run_id, response=response)
    await async_direct.on_llm_start(
        serialized={"id": ["llm"]}, prompts=[], run_id=async_run_id
    )
    await async_direct.on_llm_error(exception, run_id=async_run_id, response=response)
    dumped = dumpd(AIMessage(content="partial"))
    assert sync_direct.runs[0].outputs is not None
    assert async_direct.runs[0].outputs is not None
    assert sync_direct.runs[0].outputs["generations"][0][0]["message"] == dumped
    assert async_direct.runs[0].outputs["generations"][0][0]["message"] == dumped
