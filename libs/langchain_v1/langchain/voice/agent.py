"""Public in-process agent and session APIs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from langchain.voice.brain import GraphBrain, InputFactory, ResultFormatter
from langchain.voice.events import (
    MAX_TASK_ID_CHARS,
    ConversationTranscriptVoiceEvent,
    SessionStartedVoiceEvent,
    VoiceEvent,
    VoiceMessage,
)
from langchain.voice.prompts import build_conversation_instructions
from langchain.voice.tasks import DEFAULT_MAX_ACTIVE_TASKS, TaskManager, TaskTools
from langchain.voice.transport import LocalAudioTransport

if TYPE_CHECKING:
    from langchain.voice.audio import AudioInput, AudioOutput, StatusUI
    from langchain.voice.conversation import ConversationLayer
    from langchain.voice.transport import VoiceTransport

MAX_USER_TEXT_CHARS = 16_000
DEFAULT_EVENT_BUFFER_SIZE = 256
LIVE_AUDIO_OUTPUT_SAMPLE_RATE = 24_000
SendEvent = Callable[[VoiceEvent], Awaitable[None]]
_EVENTS_CLOSED = object()


class VoiceSession:
    """One duplex conversation and its isolated set of background tasks."""

    def __init__(
        self,
        *,
        instructions: str,
        brain: GraphBrain,
        send_event: SendEvent | None,
        max_active_tasks: int,
        event_buffer_size: int,
    ) -> None:
        """Initialize isolated state for one duplex conversation."""
        if event_buffer_size < 1:
            msg = "event_buffer_size must be positive"
            raise ValueError(msg)
        self.session_id = str(uuid4())
        self.instructions = instructions
        self._send_event = send_event
        self._events: asyncio.Queue[VoiceEvent | object] = asyncio.Queue(maxsize=event_buffer_size)
        self._start_lock = asyncio.Lock()
        self._started = False
        self._closed = False
        self._tasks = TaskManager(brain, self._on_task_event, max_active_tasks=max_active_tasks)
        self.tools = TaskTools(self._tasks)

    async def start(self) -> None:
        """Start the session and emit its ready event exactly once."""
        async with self._start_lock:
            if self._closed:
                msg = "the voice session is closed"
                raise RuntimeError(msg)
            if self._started:
                return
            self._started = True
            await self._emit(SessionStartedVoiceEvent(session_id=self.session_id))

    async def send(self, message: VoiceMessage | Mapping[str, Any]) -> str | None:
        """Send a provider-neutral message to the in-process runtime."""
        await self.start()
        parsed = VoiceMessage.from_value(message)
        if parsed.type == "input.text":
            await self.receive_text(parsed.string_field("text"))
            return None
        if parsed.type == "task.create":
            return await self.create_task(parsed.string_field("instruction"))
        if parsed.type == "task.update":
            await self.update_task(
                parsed.string_field("task_id", max_chars=MAX_TASK_ID_CHARS),
                parsed.string_field("instruction"),
            )
            return None
        if parsed.type == "task.cancel":
            await self.cancel_task(parsed.string_field("task_id", max_chars=MAX_TASK_ID_CHARS))
            return None
        if parsed.type == "session.close":
            await self.aclose()
            return None
        msg = f"unknown message type: {parsed.type}"
        raise ValueError(msg)

    async def events(self) -> AsyncIterator[VoiceEvent]:
        """Yield events until the session closes and buffered events drain."""
        await self.start()
        while True:
            if self._closed and self._events.empty():
                return
            item = await self._events.get()
            if item is _EVENTS_CLOSED:
                return
            yield cast("VoiceEvent", item)

    async def next_event(self) -> VoiceEvent:
        """Return the next event, or raise StopAsyncIteration after close."""
        await self.start()
        item = await self._events.get()
        if item is _EVENTS_CLOSED:
            raise StopAsyncIteration
        return cast("VoiceEvent", item)

    async def receive_text(self, text: str) -> str:
        """Record a final user transcript in this session."""
        await self.start()
        if self._closed:
            msg = "the voice session is closed"
            raise RuntimeError(msg)
        return await self.record_transcript("user", text)

    async def record_transcript(self, role: Literal["user", "assistant"], text: str) -> str:
        """Record final heard/said text without asking the coordinator to act."""
        await self.start()
        if self._closed:
            msg = "the voice session is closed"
            raise RuntimeError(msg)
        if role not in {"user", "assistant"}:
            msg = "role must be 'user' or 'assistant'"
            raise ValueError(msg)
        text = text.strip()
        if not text:
            msg = "text cannot be empty"
            raise ValueError(msg)
        if len(text) > MAX_USER_TEXT_CHARS:
            msg = f"text exceeds the {MAX_USER_TEXT_CHARS}-character limit"
            raise ValueError(msg)
        await self._emit(ConversationTranscriptVoiceEvent(role=role, text=text))
        return text

    async def create_task(self, instruction: str) -> str:
        """Create an independent background graph task."""
        await self.start()
        return await self.tools.create_task(instruction)

    async def update_task(self, task_id: str, instruction: str) -> None:
        """Replace the work for an existing background task."""
        await self.start()
        await self.tools.update_task(task_id, instruction)

    async def cancel_task(self, task_id: str) -> None:
        """Cancel a background task that is no longer needed."""
        await self.start()
        await self.tools.cancel_task(task_id)

    async def aclose(self) -> None:
        """Cancel unfinished work and close the session."""
        if self._closed:
            return
        self._closed = True
        await self._tasks.aclose()
        with suppress(asyncio.QueueFull):
            self._events.put_nowait(_EVENTS_CLOSED)

    async def _on_task_event(self, event: VoiceEvent) -> None:
        if self._closed:
            return
        await self._emit(event)

    async def _emit(self, event: VoiceEvent) -> None:
        await self._events.put(event)
        if self._send_event is not None:
            await self._send_event(event)


class VoiceAgent:
    """A complete duplex voice agent backed by a user-supplied LangGraph."""

    def __init__(
        self,
        *,
        instructions: str,
        brain: GraphBrain,
        conversation: ConversationLayer,
        max_active_tasks: int,
    ) -> None:
        """Initialize a reusable agent definition."""
        self.instructions = instructions
        self._brain = brain
        self.conversation = conversation
        self._max_active_tasks = max_active_tasks

    def _create_session(
        self,
        send_event: SendEvent | None = None,
        *,
        event_buffer_size: int = DEFAULT_EVENT_BUFFER_SIZE,
    ) -> VoiceSession:
        """Create isolated runtime state for one conversation-layer run."""
        return VoiceSession(
            instructions=self.instructions,
            brain=self._brain,
            send_event=send_event,
            max_active_tasks=self._max_active_tasks,
            event_buffer_size=event_buffer_size,
        )

    async def run(
        self,
        *,
        transport: VoiceTransport | None = None,
        audio_in: AudioInput | None = None,
        audio_out: AudioOutput | None = None,
        ui: StatusUI | None = None,
        project_name: str | None = None,
    ) -> None:
        """Open the configured live provider and run until the session closes.

        Args:
            transport: Duplex application media transport for this session.
            audio_in: Legacy PCM16 input. Provide together with `audio_out` when
                migrating an existing local-audio integration.
            audio_out: Legacy PCM16 output. Provide together with `audio_in`.
            ui: Optional status and transcript observer.
            project_name: Optional LangSmith tracing project.
        """
        if transport is not None and (audio_in is not None or audio_out is not None):
            msg = "provide transport or audio_in/audio_out, not both"
            raise ValueError(msg)
        if transport is None:
            if audio_in is None or audio_out is None:
                msg = "provide transport or both audio_in and audio_out"
                raise ValueError(msg)
            transport = LocalAudioTransport(audio_in, audio_out)
        if transport.output_sample_rate != LIVE_AUDIO_OUTPUT_SAMPLE_RATE:
            msg = "LangChain Voice live providers require 24 kHz audio output"
            raise ValueError(msg)
        session = self._create_session()
        try:
            await transport.start()
            await self.conversation.run(
                session,
                transport=transport,
                ui=ui,
                project_name=project_name,
            )
        finally:
            await session.aclose()
            await transport.aclose()


def create_voice_agent(
    graph: Any,
    *,
    conversation: ConversationLayer,
    input_factory: InputFactory | None = None,
    result_formatter: ResultFormatter | None = None,
    max_active_tasks: int = DEFAULT_MAX_ACTIVE_TASKS,
) -> VoiceAgent:
    """Create a complete live voice agent around a compiled LangGraph."""
    if max_active_tasks < 1:
        msg = "max_active_tasks must be positive"
        raise ValueError(msg)
    if not callable(getattr(conversation, "run", None)):
        msg = "conversation must implement an async run method"
        raise TypeError(msg)
    configured_instructions = getattr(conversation, "instructions", None)
    if not isinstance(configured_instructions, str) or not configured_instructions.strip():
        msg = "conversation.instructions must be a non-empty string"
        raise ValueError(msg)
    brain = GraphBrain(
        graph,
        input_factory=input_factory,
        result_formatter=result_formatter,
    )
    conversation_instructions = build_conversation_instructions(configured_instructions)
    return VoiceAgent(
        instructions=conversation_instructions,
        brain=brain,
        conversation=conversation,
        max_active_tasks=max_active_tasks,
    )
