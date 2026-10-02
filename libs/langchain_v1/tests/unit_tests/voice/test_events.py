from __future__ import annotations

import unittest

from langchain.voice.events import (
    ConversationMessageVoiceEvent,
    ConversationTranscriptVoiceEvent,
    ErrorVoiceEvent,
    SessionStartedVoiceEvent,
    TaskCancelledVoiceEvent,
    TaskCompletedVoiceEvent,
    TaskCreatedVoiceEvent,
    TaskFailedVoiceEvent,
    TaskStartedVoiceEvent,
    TaskUpdatedVoiceEvent,
)


class TypedVoiceEventTests(unittest.TestCase):
    def test_events_have_typed_fields_and_stable_wire_shapes(self) -> None:
        cases = [
            (
                SessionStartedVoiceEvent(session_id="session-1"),
                {
                    "type": "session.ready",
                    "session_id": "session-1",
                    "protocol_version": "0.2",
                },
            ),
            (
                ConversationTranscriptVoiceEvent(role="user", text="hello"),
                {
                    "type": "conversation.transcript",
                    "role": "user",
                    "text": "hello",
                },
            ),
            (
                ConversationMessageVoiceEvent(text="Hi there"),
                {
                    "type": "conversation.message",
                    "text": "Hi there",
                    "role": "assistant",
                },
            ),
            (
                TaskCreatedVoiceEvent(task_id="task-1", revision=1),
                {
                    "type": "task.created",
                    "task_id": "task-1",
                    "revision": 1,
                    "status": "pending",
                },
            ),
            (
                TaskStartedVoiceEvent(task_id="task-1", revision=1),
                {
                    "type": "task.started",
                    "task_id": "task-1",
                    "revision": 1,
                    "status": "running",
                },
            ),
            (
                TaskUpdatedVoiceEvent(task_id="task-1", revision=2),
                {
                    "type": "task.updated",
                    "task_id": "task-1",
                    "revision": 2,
                    "status": "pending",
                },
            ),
            (
                TaskCompletedVoiceEvent(task_id="task-1", revision=2, result="sunny"),
                {
                    "type": "task.completed",
                    "task_id": "task-1",
                    "revision": 2,
                    "result": "sunny",
                    "status": "completed",
                },
            ),
            (
                TaskFailedVoiceEvent(task_id="task-1", revision=2, error="could not finish"),
                {
                    "type": "task.failed",
                    "task_id": "task-1",
                    "revision": 2,
                    "error": "could not finish",
                    "status": "failed",
                },
            ),
            (
                TaskCancelledVoiceEvent(task_id="task-1", revision=3),
                {
                    "type": "task.cancelled",
                    "task_id": "task-1",
                    "revision": 3,
                    "status": "cancelled",
                },
            ),
            (
                ErrorVoiceEvent(code="invalid_message", message="bad input"),
                {
                    "type": "error",
                    "code": "invalid_message",
                    "message": "bad input",
                },
            ),
        ]

        for event, expected in cases:
            with self.subTest(event=event.type):
                assert not hasattr(event, "data")
                assert event.as_dict() == expected


if __name__ == "__main__":
    unittest.main()
