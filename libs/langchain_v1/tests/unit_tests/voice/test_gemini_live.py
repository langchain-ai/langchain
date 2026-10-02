from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from langchain.voice.diagnostics import make_debug_log as _make_debug_log
from langchain.voice.events import TaskCompletedVoiceEvent
from langchain.voice.providers.gemini_live import (
    _deliver_terminal_task_results,
    _function_response,
    _TerminalTaskResultDispatcher,
    gemini_task_tool,
)


class FakeFunctionDeclaration:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeTool:
    def __init__(self, *, function_declarations: list[Any]) -> None:
        self.function_declarations = function_declarations


class FakeFunctionResponse:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeTypes:
    class Behavior:
        NON_BLOCKING = "non-blocking"
        BLOCKING = "blocking"

    class FunctionResponseScheduling:
        WHEN_IDLE = "when-idle"
        SILENT = "silent"

    FunctionDeclaration = FakeFunctionDeclaration
    FunctionResponse = FakeFunctionResponse
    Tool = FakeTool


class FakeConnection:
    def __init__(self) -> None:
        self.sent: list[list[Any]] = []
        self.changed = asyncio.Event()

    async def send_tool_response(self, *, function_responses: list[Any]) -> None:
        self.sent.append(function_responses)
        self.changed.set()

    async def wait_for_send_count(self, count: int) -> None:
        async def wait_for_change() -> None:
            while len(self.sent) < count:
                self.changed.clear()
                if len(self.sent) < count:
                    await self.changed.wait()

        await asyncio.wait_for(wait_for_change(), 1)


class GeminiLiveTests(unittest.IsolatedAsyncioTestCase):
    def test_debug_logging_is_opt_in(self) -> None:
        class FakeUI:
            def __init__(self) -> None:
                self.logs: list[str] = []

            def log(self, message: str) -> None:
                self.logs.append(message)

        ui = FakeUI()
        with patch.dict(os.environ, {"LANGCHAIN_VOICE_DEBUG": "0"}):
            _make_debug_log(ui)("disabled")
        with patch.dict(os.environ, {"LANGCHAIN_VOICE_DEBUG": "1"}):
            _make_debug_log(ui)("enabled")

        assert ui.logs == ["[langchain.voice:debug] enabled"]

    def test_create_task_is_a_non_blocking_generator(self) -> None:
        declarations = gemini_task_tool(FakeTypes).function_declarations
        behaviors = {declaration.name: declaration.behavior for declaration in declarations}

        assert behaviors["create_task"] == FakeTypes.Behavior.NON_BLOCKING
        assert behaviors["update_task"] == FakeTypes.Behavior.BLOCKING
        assert behaviors["cancel_task"] == FakeTypes.Behavior.BLOCKING

    def test_function_response_preserves_generator_schedule(self) -> None:
        call = SimpleNamespace(id="call-1", name="create_task")

        started = _function_response(
            FakeTypes,
            call,
            {"task_id": "task-1", "status": "started"},
            will_continue=True,
            scheduling=FakeTypes.FunctionResponseScheduling.WHEN_IDLE,
        )
        closed = _function_response(
            FakeTypes,
            call,
            {},
            will_continue=False,
            scheduling=FakeTypes.FunctionResponseScheduling.SILENT,
        )

        assert started.id == closed.id
        assert started.will_continue
        assert not closed.will_continue
        assert closed.scheduling == FakeTypes.FunctionResponseScheduling.SILENT

    async def test_terminal_results_generate_once_then_close_silently(
        self,
    ) -> None:
        connection = FakeConnection()
        calls = [
            SimpleNamespace(id="call-1", name="create_task"),
            SimpleNamespace(id="call-2", name="create_task"),
        ]
        events = [
            TaskCompletedVoiceEvent(task_id="task-1", revision=1, result="sunny"),
            TaskCompletedVoiceEvent(task_id="task-2", revision=1, result="windy"),
        ]

        await _deliver_terminal_task_results(
            connection, FakeTypes, list(zip(calls, events, strict=True))
        )

        assert len(connection.sent) == 2
        results, closures = connection.sent
        assert [response.scheduling for response in results] == [
            FakeTypes.FunctionResponseScheduling.SILENT,
            FakeTypes.FunctionResponseScheduling.WHEN_IDLE,
        ]
        assert all(response.will_continue for response in results)
        assert [response.response["output"]["result"] for response in results] == ["sunny", "windy"]
        assert all(
            response.scheduling == FakeTypes.FunctionResponseScheduling.SILENT
            for response in closures
        )
        assert all(not response.will_continue for response in closures)
        assert all(response.response == {} for response in closures)

    async def test_dispatcher_holds_later_result_until_turn_complete(
        self,
    ) -> None:
        connection = FakeConnection()
        dispatcher = _TerminalTaskResultDispatcher(connection, FakeTypes)
        first_call = SimpleNamespace(id="call-1", name="create_task")
        second_call = SimpleNamespace(id="call-2", name="create_task")
        runner = asyncio.create_task(dispatcher.run())
        self.addAsyncCleanup(self._cancel_task, runner)

        await dispatcher.enqueue(
            first_call,
            TaskCompletedVoiceEvent(task_id="task-1", revision=1, result="sunny"),
        )
        await connection.wait_for_send_count(2)
        await dispatcher.enqueue(
            second_call,
            TaskCompletedVoiceEvent(task_id="task-2", revision=1, result="windy"),
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert len(connection.sent) == 2

        dispatcher.mark_idle()
        await connection.wait_for_send_count(4)
        assert connection.sent[2][0].response["output"]["result"] == "windy"

    async def test_dispatcher_coalesces_results_waiting_for_idle(self) -> None:
        connection = FakeConnection()
        debug_logs: list[str] = []
        dispatcher = _TerminalTaskResultDispatcher(connection, FakeTypes, debug_logs.append)
        dispatcher.mark_busy()
        runner = asyncio.create_task(dispatcher.run())
        self.addAsyncCleanup(self._cancel_task, runner)

        await dispatcher.enqueue(
            SimpleNamespace(id="call-1", name="create_task"),
            TaskCompletedVoiceEvent(task_id="task-1", revision=1, result="sunny"),
        )
        await dispatcher.enqueue(
            SimpleNamespace(id="call-2", name="create_task"),
            TaskCompletedVoiceEvent(task_id="task-2", revision=1, result="windy"),
        )
        await asyncio.sleep(0)
        assert connection.sent == []

        dispatcher.mark_idle()
        await connection.wait_for_send_count(2)

        results, closures = connection.sent
        assert len(results) == 2
        assert len(closures) == 2
        assert (
            sum(
                response.scheduling == FakeTypes.FunctionResponseScheduling.WHEN_IDLE
                for response in results
            )
            == 1
        )
        dispatcher.mark_idle(reason="test_turn_complete")
        joined_logs = "\n".join(debug_logs)
        assert "result_queued" in joined_logs
        assert "result_batch_dispatch" in joined_logs
        assert "count=2" in joined_logs
        assert "generation_triggers=1" in joined_logs
        assert "result_followup_complete" in joined_logs
        assert "sunny" not in joined_logs
        assert "windy" not in joined_logs

    async def _cancel_task(self, task: asyncio.Task[None]) -> None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
