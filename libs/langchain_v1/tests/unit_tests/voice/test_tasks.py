from __future__ import annotations

import asyncio
import unittest
from typing import TYPE_CHECKING, Any

import pytest

from langchain.voice.brain import GraphBrain
from langchain.voice.events import TaskCompletedVoiceEvent, VoiceEvent
from langchain.voice.tasks import TaskManager, TaskNotFoundError, TaskStatus

if TYPE_CHECKING:
    from collections.abc import Callable


async def wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    async def poll() -> None:
        while not predicate():  # noqa: ASYNC110 - cooperative test polling
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout)


class ParallelGraph:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, str]:
        del config
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return {"response": state["messages"][0]["content"]}


class UpdatingGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.first_started = asyncio.Event()
        self.first_cancelled = asyncio.Event()

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, str]:
        assert config is not None
        instruction = state["messages"][0]["content"]
        self.calls.append((instruction, config["configurable"]["thread_id"]))
        if len(self.calls) == 1:
            self.first_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.first_cancelled.set()
                raise
        return {"response": instruction}


class TaskManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_tasks_run_in_parallel(self) -> None:
        events: list[VoiceEvent] = []

        async def sink(event: VoiceEvent) -> None:
            events.append(event)

        graph = ParallelGraph()
        manager = TaskManager(GraphBrain(graph), sink)
        first = await manager.create_task("one")
        second = await manager.create_task("two")
        await wait_until(lambda: len([e for e in events if e.type == "task.completed"]) == 2)

        assert first != second
        assert graph.max_active == 2
        await manager.aclose()

    async def test_update_cancels_old_run_and_reuses_thread(self) -> None:
        events: list[VoiceEvent] = []

        async def sink(event: VoiceEvent) -> None:
            events.append(event)

        graph = UpdatingGraph()
        manager = TaskManager(GraphBrain(graph), sink)
        task_id = await manager.create_task("all London airports")
        await graph.first_started.wait()

        await manager.update_task(task_id, "Heathrow only")
        await wait_until(lambda: any(e.type == "task.completed" for e in events))

        assert graph.first_cancelled.is_set()
        assert graph.calls[0][1] == graph.calls[1][1]
        completed = [e for e in events if isinstance(e, TaskCompletedVoiceEvent)]
        assert len(completed) == 1
        assert completed[0].result == "Heathrow only"
        assert completed[0].revision == 2
        await manager.aclose()

    async def test_cancel_is_terminal(self) -> None:
        events: list[VoiceEvent] = []

        async def sink(event: VoiceEvent) -> None:
            events.append(event)

        graph = UpdatingGraph()
        manager = TaskManager(GraphBrain(graph), sink)
        task_id = await manager.create_task("slow task")
        await graph.first_started.wait()
        await manager.cancel_task(task_id)

        assert await manager.get_status(task_id) == TaskStatus.CANCELLED
        assert any(e.type == "task.cancelled" for e in events)
        await manager.aclose()

    async def test_task_ids_are_scoped_to_manager(self) -> None:
        async def sink(event: VoiceEvent) -> None:
            pass

        first = TaskManager(GraphBrain(ParallelGraph()), sink)
        second = TaskManager(GraphBrain(ParallelGraph()), sink)
        task_id = await first.create_task("private")

        with pytest.raises(TaskNotFoundError):
            await second.cancel_task(task_id)

        await first.aclose()
        await second.aclose()


if __name__ == "__main__":
    unittest.main()
