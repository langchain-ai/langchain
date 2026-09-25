from __future__ import annotations

import asyncio
import unittest
from typing import TYPE_CHECKING, Any, cast

import pytest

from langchain.voice import create_voice_agent
from langchain.voice.events import SessionStartedVoiceEvent, TaskCompletedVoiceEvent

if TYPE_CHECKING:
    from langchain.voice.agent import VoiceSession
    from langchain.voice.audio import StatusUI
    from langchain.voice.transport import VoiceTransport


class BarrierGraph:
    def __init__(self, expected_calls: int) -> None:
        self.expected_calls = expected_calls
        self.active = 0
        self.max_active = 0
        self.all_started = asyncio.Event()
        self.calls: list[tuple[str, str]] = []

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, str]:
        assert config is not None
        instruction = state["messages"][0]["content"]
        thread_id = config["configurable"]["thread_id"]
        self.calls.append((instruction, thread_id))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active == self.expected_calls:
            self.all_started.set()
        await self.all_started.wait()
        await asyncio.sleep(0)
        self.active -= 1
        return {"response": instruction}


class LabeledTransport:
    output_sample_rate = 24_000
    output_active = False

    def __init__(self, label: str) -> None:
        self.label = label
        self.closed = False

    async def start(self) -> None:
        return

    async def aclose(self) -> None:
        self.closed = True


class ParallelConversationLayer:
    instructions = "Be concise."

    def __init__(self) -> None:
        self.sessions: list[VoiceSession] = []
        self.task_ids: set[str] = set()
        self.results: dict[str, str] = {}

    async def run(
        self,
        session: VoiceSession,
        *,
        transport: VoiceTransport,
        ui: StatusUI | None,
        project_name: str | None,
    ) -> None:
        del ui, project_name
        self.sessions.append(session)
        events = session.events()
        started = await anext(events)
        if not isinstance(started, SessionStartedVoiceEvent):
            msg = "session did not start with a typed event"
            raise TypeError(msg)

        label = cast("LabeledTransport", transport).label
        task_id = await session.create_task(label)
        self.task_ids.add(task_id)
        async for event in events:
            if isinstance(event, TaskCompletedVoiceEvent):
                self.results[label] = event.result
                return


class ParallelSessionIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_five_parallel_runs_have_isolated_sessions_and_threads(
        self,
    ) -> None:
        session_count = 5
        graph = BarrierGraph(expected_calls=session_count)
        conversation = ParallelConversationLayer()
        agent = create_voice_agent(graph, conversation=conversation)
        labels = [f"request-{index}" for index in range(session_count)]

        await asyncio.gather(
            *(
                agent.run(
                    transport=cast("Any", LabeledTransport(label)),
                )
                for label in labels
            )
        )

        assert graph.max_active == session_count
        assert len(conversation.sessions) == session_count
        assert len({session.session_id for session in conversation.sessions}) == session_count
        assert len(conversation.task_ids) == session_count
        assert len({thread_id for _, thread_id in graph.calls}) == session_count
        assert conversation.results == {label: label for label in labels}
        for session in conversation.sessions:
            with pytest.raises(RuntimeError, match="closed"):
                await session.create_task("work after shutdown")


if __name__ == "__main__":
    unittest.main()
