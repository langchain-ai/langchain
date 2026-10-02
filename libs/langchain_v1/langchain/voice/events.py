"""Typed, provider-neutral messages and events used by LangChain Voice."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal

MAX_MESSAGE_TEXT_CHARS = 16_000
MAX_TASK_ID_CHARS = 128


@dataclass(frozen=True, slots=True)
class VoiceEvent:
    """Base class for the typed events emitted by a voice session."""

    type: ClassVar[str]

    def as_dict(self) -> dict[str, Any]:
        """Serialize this event to its transport-neutral mapping."""
        return {
            "type": self.type,
            **{key: value for key, value in asdict(self).items() if value is not None},
        }


@dataclass(frozen=True, slots=True)
class SessionStartedVoiceEvent(VoiceEvent):
    """Signal that an isolated voice session is ready."""

    type: ClassVar[Literal["session.ready"]] = "session.ready"
    session_id: str
    protocol_version: str = "0.2"


@dataclass(frozen=True, slots=True)
class ConversationTranscriptVoiceEvent(VoiceEvent):
    """Record final text that the user heard or said."""

    type: ClassVar[Literal["conversation.transcript"]] = "conversation.transcript"
    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True, slots=True)
class ConversationMessageVoiceEvent(VoiceEvent):
    """Request delivery of an assistant text message."""

    type: ClassVar[Literal["conversation.message"]] = "conversation.message"
    text: str
    role: Literal["assistant"] = "assistant"
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class TaskCreatedVoiceEvent(VoiceEvent):
    """Signal that a background task was accepted."""

    type: ClassVar[Literal["task.created"]] = "task.created"
    task_id: str
    revision: int
    status: Literal["pending"] = "pending"


@dataclass(frozen=True, slots=True)
class TaskStartedVoiceEvent(VoiceEvent):
    """Signal that a task revision started running."""

    type: ClassVar[Literal["task.started"]] = "task.started"
    task_id: str
    revision: int
    status: Literal["running"] = "running"


@dataclass(frozen=True, slots=True)
class TaskUpdatedVoiceEvent(VoiceEvent):
    """Signal that a task was updated with a new revision."""

    type: ClassVar[Literal["task.updated"]] = "task.updated"
    task_id: str
    revision: int
    status: Literal["pending"] = "pending"


@dataclass(frozen=True, slots=True)
class TaskCompletedVoiceEvent(VoiceEvent):
    """Carry the successful result of a task revision."""

    type: ClassVar[Literal["task.completed"]] = "task.completed"
    task_id: str
    revision: int
    result: str
    status: Literal["completed"] = "completed"


@dataclass(frozen=True, slots=True)
class TaskFailedVoiceEvent(VoiceEvent):
    """Carry a safe failure description for a task revision."""

    type: ClassVar[Literal["task.failed"]] = "task.failed"
    task_id: str
    revision: int
    error: str
    status: Literal["failed"] = "failed"


@dataclass(frozen=True, slots=True)
class TaskCancelledVoiceEvent(VoiceEvent):
    """Signal that a task revision was cancelled."""

    type: ClassVar[Literal["task.cancelled"]] = "task.cancelled"
    task_id: str
    revision: int
    status: Literal["cancelled"] = "cancelled"


@dataclass(frozen=True, slots=True)
class ErrorVoiceEvent(VoiceEvent):
    """Carry a session-level protocol or runtime error."""

    type: ClassVar[Literal["error"]] = "error"
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class VoiceMessage:
    """A provider-neutral command sent to an in-process voice session."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: VoiceMessage | Mapping[str, Any]) -> VoiceMessage:
        """Normalize a typed message or mapping into a `VoiceMessage`."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            msg = "message must be a VoiceMessage or mapping"
            raise TypeError(msg)
        message_type = value.get("type")
        if not isinstance(message_type, str) or not message_type:
            msg = "message type must be a non-empty string"
            raise ValueError(msg)
        data = {key: item for key, item in value.items() if key != "type"}
        return cls(message_type, data)

    def string_field(self, name: str, *, max_chars: int = MAX_MESSAGE_TEXT_CHARS) -> str:
        """Return one required, normalized, bounded string field."""
        value = self.data.get(name)
        if not isinstance(value, str) or not value.strip():
            msg = f"{name} must be a non-empty string"
            raise ValueError(msg)
        normalized = value.strip()
        if len(normalized) > max_chars:
            msg = f"{name} exceeds the {max_chars}-character limit"
            raise ValueError(msg)
        return normalized


@dataclass(frozen=True, slots=True)
class ConversationReply:
    """Text the conversation layer wants the user to hear."""

    text: str
    task_id: str | None = None

    def as_event(self) -> ConversationMessageVoiceEvent:
        """Convert this reply to the corresponding session event."""
        return ConversationMessageVoiceEvent(text=self.text, task_id=self.task_id)
