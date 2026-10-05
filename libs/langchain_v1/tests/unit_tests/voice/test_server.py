from __future__ import annotations

import asyncio
import json
import unittest
from typing import TYPE_CHECKING, Any

import pytest

from langchain.voice import create_voice_agent
from langchain.voice.events import ConversationReply, TaskCompletedVoiceEvent, VoiceEvent
from langchain.voice.transports.websocket import ProtocolError, WebSocketServer

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain.voice.agent import VoiceSession
    from langchain.voice.tasks import TaskTools


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def send(self, message: dict[str, Any]) -> None:
        self.calls.append(message)


class FakeWebSocket:
    def __init__(
        self,
        incoming: list[str | bytes],
        stop_when: Callable[[list[dict[str, Any]]], bool],
    ) -> None:
        self._incoming = iter(incoming)
        self._stop_when = stop_when
        self._done = asyncio.Event()
        self.sent: list[dict[str, Any]] = []
        self.closed: tuple[int, str] | None = None

    def __aiter__(self) -> FakeWebSocket:
        return self

    async def __anext__(self) -> str | bytes:
        try:
            return next(self._incoming)
        except StopIteration:
            await self._done.wait()
            raise StopAsyncIteration from None

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))
        if self._stop_when(self.sent):
            self._done.set()

    async def close(self, *, code: int, reason: str) -> None:
        self.closed = (code, reason)
        self._done.set()


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_dispatches_task_controls(self) -> None:
        server = WebSocketServer(agent=None)
        session = FakeSession()

        await server._dispatch(
            session,
            '{"type":"task.update","task_id":"task-1","instruction":"Heathrow"}',
        )

        assert session.calls == [
            {
                "type": "task.update",
                "task_id": "task-1",
                "instruction": "Heathrow",
            }
        ]

    async def test_rejects_invalid_json(self) -> None:
        server = WebSocketServer(agent=None)
        with pytest.raises(ProtocolError, match="valid JSON"):
            await server._dispatch(FakeSession(), "not json")

    async def test_rejects_unknown_message(self) -> None:
        server = WebSocketServer(agent=None)
        with pytest.raises(ProtocolError, match="Unknown message type"):
            await server._dispatch(FakeSession(), '{"type":"audio.magic"}')

    async def test_session_close_requests_clean_shutdown(self) -> None:
        server = WebSocketServer(agent=None)
        assert await server._dispatch(FakeSession(), '{"type":"session.close"}')


class ImmediateGraph:
    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, str]:
        del state, config
        return {"response": "websocket result"}


class FakeLiveConversation:
    instructions = "Be brief."

    async def run(self, session: VoiceSession, **kwargs: Any) -> None:
        del session, kwargs


class TaskingConversation:
    def __init__(self, instructions: str) -> None:
        self.instructions = instructions

    async def on_user_text(self, text: str, tools: TaskTools) -> ConversationReply:
        task_id = await tools.create_task(text)
        return ConversationReply("I'm on it.", task_id)

    async def on_task_event(self, event: VoiceEvent, tools: TaskTools) -> ConversationReply | None:
        del tools
        if isinstance(event, TaskCompletedVoiceEvent):
            return ConversationReply(event.result, event.task_id)
        return None


class WebSocketConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_conversation_round_trip(self) -> None:
        agent = create_voice_agent(
            ImmediateGraph(),
            conversation=FakeLiveConversation(),
        )
        transport = WebSocketServer(agent, conversation_factory=TaskingConversation)

        socket = FakeWebSocket(
            [json.dumps({"type": "input.text", "text": "do the work"})],
            lambda events: sum(event["type"] == "conversation.message" for event in events) == 2,
        )

        await transport._handle_connection(socket)

        assert socket.sent[0]["type"] == "session.ready"
        messages = [
            event["text"] for event in socket.sent if event["type"] == "conversation.message"
        ]
        assert messages == ["I'm on it.", "websocket result"]

    async def test_binary_frames_get_a_stable_error(self) -> None:
        agent = create_voice_agent(
            ImmediateGraph(),
            conversation=FakeLiveConversation(),
        )
        transport = WebSocketServer(agent, conversation_factory=TaskingConversation)

        socket = FakeWebSocket(
            [b"audio"],
            lambda events: any(event["type"] == "error" for event in events),
        )

        await transport._handle_connection(socket)

        error = next(event for event in socket.sent if event["type"] == "error")
        assert error["code"] == "binary_not_supported"


if __name__ == "__main__":
    unittest.main()
