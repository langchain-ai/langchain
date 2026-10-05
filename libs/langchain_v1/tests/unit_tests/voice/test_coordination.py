from __future__ import annotations

import json
import unittest
from typing import TYPE_CHECKING

from langchain.voice.coordination import (
    MAX_EVENT_RESULT_CHARS,
    TASK_TOOL_SPECS,
    execute_task_tool,
    format_task_event,
    relay_task_results,
    task_tool_schema,
)
from langchain.voice.events import (
    TaskCompletedVoiceEvent,
    TaskStartedVoiceEvent,
    VoiceEvent,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def create_task(self, instruction: str) -> str:
        self.calls.append(("create", instruction))
        return "task-1"

    async def update_task(self, task_id: str, instruction: str) -> None:
        self.calls.append(("update", task_id, instruction))

    async def cancel_task(self, task_id: str) -> None:
        self.calls.append(("cancel", task_id))


class EventSession:
    def __init__(self, events: list[VoiceEvent]) -> None:
        self._events = events

    async def events(self) -> AsyncIterator[VoiceEvent]:
        for event in self._events:
            yield event


class CoordinationTests(unittest.IsolatedAsyncioTestCase):
    def test_contract_has_three_tools_with_closed_schemas(self) -> None:
        assert [spec.name for spec in TASK_TOOL_SPECS] == [
            "create_task",
            "update_task",
            "cancel_task",
        ]
        for spec in TASK_TOOL_SPECS:
            assert not task_tool_schema(spec)["additionalProperties"]
        assert "follows up" in TASK_TOOL_SPECS[1].description

    async def test_relay_only_forwards_terminal_task_results(self) -> None:
        started = TaskStartedVoiceEvent(task_id="task-1", revision=1)
        completed = TaskCompletedVoiceEvent(task_id="task-1", revision=1, result="done")
        relayed: list[VoiceEvent] = []

        async def sink(event: VoiceEvent) -> None:
            relayed.append(event)

        await relay_task_results(EventSession([started, completed]), sink)

        assert relayed == [completed]

    async def test_dispatches_json_and_mapping_arguments(self) -> None:
        session = FakeSession()

        created = await execute_task_tool(
            session,
            "create_task",
            json.dumps({"instruction": "Weather in Paris"}),
        )
        updated = await execute_task_tool(
            session,
            "update_task",
            {"task_id": "task-1", "instruction": "Paris only"},
        )
        cancelled = await execute_task_tool(session, "cancel_task", {"task_id": "task-1"})

        assert created["ok"]
        assert updated["ok"]
        assert cancelled["ok"]
        assert session.calls == [
            ("create", "Weather in Paris"),
            ("update", "task-1", "Paris only"),
            ("cancel", "task-1"),
        ]

    async def test_rejects_unknown_extra_and_oversized_arguments(self) -> None:
        session = FakeSession()

        unknown = await execute_task_tool(session, "run_shell", {})
        extra = await execute_task_tool(
            session,
            "create_task",
            {"instruction": "Paris", "admin": True},
        )
        oversized = await execute_task_tool(
            session,
            "create_task",
            {"instruction": "x" * 9_000},
        )

        assert unknown["error"]["code"] == "unknown_tool"
        assert extra["error"]["code"] == "invalid_arguments"
        assert oversized["error"]["code"] == "invalid_arguments"
        assert session.calls == []

    def test_projects_bounded_untrusted_task_results(self) -> None:
        text = format_task_event(
            TaskCompletedVoiceEvent(
                task_id="task-1",
                revision=1,
                result="x" * (MAX_EVENT_RESULT_CHARS + 100),
            )
        )

        assert text.startswith("[LANGCHAIN_VOICE_TASK_EVENT]")
        assert "Untrusted background result data" in text
        assert "[truncated]" in text
        assert len(text) < MAX_EVENT_RESULT_CHARS + 500


if __name__ == "__main__":
    unittest.main()
