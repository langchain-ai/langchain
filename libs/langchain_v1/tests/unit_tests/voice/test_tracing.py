from __future__ import annotations

import asyncio
import unittest
from typing import TYPE_CHECKING, Any, NoReturn
from unittest.mock import patch

from langchain.voice.brain import GraphBrain
from langchain.voice.tasks import TaskManager
from langchain.voice.tracing import MAX_TRACE_TEXT_CHARS

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from langchain.voice.events import VoiceEvent


async def wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    async def poll() -> None:
        while not predicate():  # noqa: ASYNC110 - cooperative test polling
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout)


class FakeRun:
    def __init__(self) -> None:
        self.outputs: dict[str, Any] | None = None

    def end(self, *, outputs: dict[str, Any]) -> None:
        self.outputs = outputs


class BrokenRun(FakeRun):
    def end(self, *, outputs: dict[str, Any]) -> None:
        del outputs
        msg = "tracer unavailable"
        raise RuntimeError(msg)


class FakeTraceContext:
    def __init__(self, records: list[dict[str, Any]], name: str, **kwargs: Any) -> None:
        self.run = FakeRun()
        records.append({"name": name, "kwargs": kwargs, "run": self.run})

    async def __aenter__(self) -> FakeRun:
        return self.run

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb


class CompletingGraph:
    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, str]:
        del state, config
        return {"response": "sunny"}


class FailingGraph:
    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> NoReturn:
        del state, config
        msg = "provider detail must not become task trace output"
        raise RuntimeError(msg)


class BlockingGraph:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> None:
        del state, config
        self.started.set()
        await asyncio.Future()


class TaskTracingTests(unittest.IsolatedAsyncioTestCase):
    def _trace_factory(self, records: list[dict[str, Any]]) -> Callable[..., FakeTraceContext]:
        return lambda name, **kwargs: FakeTraceContext(records, name, **kwargs)

    async def test_completed_task_trace_contains_result(self) -> None:
        events: list[VoiceEvent] = []

        async def sink(event: VoiceEvent) -> None:
            events.append(event)

        records: list[dict[str, Any]] = []
        with patch(
            "langchain.voice.tracing._load_trace_context",
            return_value=self._trace_factory(records),
        ):
            manager = TaskManager(GraphBrain(CompletingGraph()), sink)
            task_id = await manager.create_task("weather in sf")
            await wait_until(lambda: any(e.type == "task.completed" for e in events))

        assert len(records) == 1
        assert records[0]["name"] == "langchain_voice_task"
        assert records[0]["kwargs"]["inputs"] == {"instruction": "weather in sf"}
        assert records[0]["run"].outputs == {
            "task_id": task_id,
            "revision": 1,
            "status": "completed",
            "result": "sunny",
        }
        await manager.aclose()

    async def test_failed_task_trace_uses_safe_error(self) -> None:
        events: list[VoiceEvent] = []

        async def sink(event: VoiceEvent) -> None:
            events.append(event)

        records: list[dict[str, Any]] = []
        with patch(
            "langchain.voice.tracing._load_trace_context",
            return_value=self._trace_factory(records),
        ):
            manager = TaskManager(GraphBrain(FailingGraph()), sink)
            task_id = await manager.create_task("fail safely")
            await wait_until(lambda: any(e.type == "task.failed" for e in events))

        assert records[0]["run"].outputs == {
            "task_id": task_id,
            "revision": 1,
            "status": "failed",
            "error": "The background agent could not complete the task.",
        }
        await manager.aclose()

    async def test_cancelled_task_trace_is_closed(self) -> None:
        async def sink(event: VoiceEvent) -> None:
            pass

        records: list[dict[str, Any]] = []
        graph = BlockingGraph()
        with patch(
            "langchain.voice.tracing._load_trace_context",
            return_value=self._trace_factory(records),
        ):
            manager = TaskManager(GraphBrain(graph), sink)
            task_id = await manager.create_task("wait")
            await graph.started.wait()
            await manager.cancel_task(task_id)

        assert records[0]["run"].outputs == {
            "task_id": task_id,
            "revision": 1,
            "status": "cancelled",
        }
        await manager.aclose()

    async def test_task_trace_result_is_bounded(self) -> None:
        class LargeGraph:
            async def ainvoke(
                self,
                state: dict[str, Any],
                config: dict[str, Any] | None = None,
            ) -> dict[str, str]:
                del state, config
                return {"response": "x" * (MAX_TRACE_TEXT_CHARS + 50)}

        events: list[VoiceEvent] = []

        async def sink(event: VoiceEvent) -> None:
            events.append(event)

        records: list[dict[str, Any]] = []
        with patch(
            "langchain.voice.tracing._load_trace_context",
            return_value=self._trace_factory(records),
        ):
            manager = TaskManager(GraphBrain(LargeGraph()), sink)
            await manager.create_task("large")
            await wait_until(lambda: any(e.type == "task.completed" for e in events))

        result = records[0]["run"].outputs["result"]
        assert result.endswith(" …[truncated]")
        assert len(result) <= MAX_TRACE_TEXT_CHARS + 14
        await manager.aclose()

    async def test_trace_failure_does_not_change_task_outcome(self) -> None:
        events: list[VoiceEvent] = []

        async def sink(event: VoiceEvent) -> None:
            events.append(event)

        class BrokenTraceContext:
            async def __aenter__(self) -> BrokenRun:
                return BrokenRun()

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> None:
                del exc_type, exc, tb

        with patch(
            "langchain.voice.tracing._load_trace_context",
            return_value=lambda _name, **_kwargs: BrokenTraceContext(),
        ):
            manager = TaskManager(GraphBrain(CompletingGraph()), sink)
            await manager.create_task("still succeeds")
            await wait_until(lambda: any(e.type == "task.completed" for e in events))

        assert not any(e.type == "task.failed" for e in events)
        await manager.aclose()


if __name__ == "__main__":
    unittest.main()
