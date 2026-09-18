"""OpenAI Realtime conversation runtime owned entirely by LangChain Voice."""

from __future__ import annotations

import asyncio
import base64
import heapq
import json
import os
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain.voice.audio import (
    NullUI,
    StatusUI,
    frame_level,
    resample_pcm16,
)
from langchain.voice.coordination import (
    TASK_TOOL_SPECS,
    TerminalTaskEvent,
    execute_task_tool,
    format_task_event,
    relay_task_results,
    task_tool_schema,
)
from langchain.voice.transport import (
    AudioFormat,
    AudioFrame,
    AudioPlayed,
    AudioReceived,
    TransportDisconnected,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain.voice.agent import VoiceSession
    from langchain.voice.transport import VoiceTransport

DEFAULT_MODEL = "gpt-realtime-2"
DEFAULT_VOICE = "alloy"
SAMPLE_RATE = 24_000
USER_RESPONSE = 0
TOOL_RESPONSE = 1
BACKGROUND_RESPONSE = 2
MAX_PENDING_RESPONSES = 64


@dataclass(frozen=True, slots=True)
class OpenAIRealtimeConversationLayer:
    """Configured OpenAI Realtime conversation runtime."""

    instructions: str
    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE

    def __post_init__(self) -> None:
        """Validate and normalize the provider configuration."""
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            msg = "instructions must be a non-empty string"
            raise ValueError(msg)
        if not isinstance(self.model, str) or not self.model.strip():
            msg = "model must be a non-empty string"
            raise ValueError(msg)
        if not isinstance(self.voice, str) or not self.voice.strip():
            msg = "voice must be a non-empty string"
            raise ValueError(msg)
        object.__setattr__(self, "instructions", self.instructions.strip())
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "voice", self.voice.strip())

    async def run(
        self,
        session: VoiceSession,
        *,
        transport: VoiceTransport,
        ui: StatusUI | None,
        project_name: str | None,
    ) -> None:
        """Run one Realtime conversation using an isolated voice session."""
        await _run(
            session,
            transport=transport,
            ui=ui,
            project_name=project_name,
            model=self.model,
            voice=self.voice,
        )


def openai_task_tools() -> list[dict[str, Any]]:
    """Build provider schemas for LangChain Voice's fixed coordination contract."""
    return [
        {
            "type": "function",
            "name": spec.name,
            "description": spec.description,
            "parameters": task_tool_schema(spec),
        }
        for spec in TASK_TOOL_SPECS
    ]


def session_config(instructions: str, *, voice: str = DEFAULT_VOICE) -> dict[str, Any]:
    """Build an interruptible OpenAI Realtime session configuration."""
    return {
        "type": "realtime",
        "instructions": instructions,
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                "transcription": {"model": "gpt-4o-mini-transcribe"},
                "noise_reduction": {"type": "near_field"},
                "turn_detection": {
                    "type": "server_vad",
                    "create_response": False,
                    "interrupt_response": True,
                },
            },
            "output": {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                "voice": voice,
            },
        },
        "tools": openai_task_tools(),
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }


BeforeResponse = Callable[[], Awaitable[bool | None]]


class ResponseScheduler:
    """Serialize responses while prioritizing live user turns."""

    def __init__(self, connection: Any, transport: VoiceTransport) -> None:
        """Initialize the scheduler for one provider connection."""
        self._connection = connection
        self._transport = transport
        self._condition = asyncio.Condition()
        self._items: list[tuple[int, int, BeforeResponse | None]] = []
        self._sequence = 0
        self._response_idle = asyncio.Event()
        self._response_idle.set()
        self._user_idle = asyncio.Event()
        self._user_idle.set()
        self._closed = False
        self._worker: asyncio.Task[None] | None = None

    @property
    def response_idle(self) -> bool:
        """Return whether the provider has no response in progress."""
        return self._response_idle.is_set()

    def start(self) -> None:
        """Start the background response scheduler."""
        self._worker = asyncio.create_task(self._run(), name="langchain-voice-responses")

    def user_speech_started(self) -> None:
        """Prevent queued responses from starting while the user is speaking."""
        self._user_idle.clear()

    def user_speech_finished(self) -> None:
        """Allow the next prioritized response to start."""
        self._user_idle.set()

    async def request(self, priority: int, before_response: BeforeResponse | None = None) -> None:
        """Queue a response request at the given priority."""
        async with self._condition:
            self._sequence += 1
            if len(self._items) >= MAX_PENDING_RESPONSES:
                worst_priority = max(item[0] for item in self._items)
                if worst_priority < priority or (
                    worst_priority == priority and priority == BACKGROUND_RESPONSE
                ):
                    return
                worst_index = min(
                    (index for index, item in enumerate(self._items) if item[0] == worst_priority),
                    key=lambda index: self._items[index][1],
                )
                self._items.pop(worst_index)
                heapq.heapify(self._items)
            heapq.heappush(self._items, (priority, self._sequence, before_response))
            self._condition.notify()

    def mark_done(self) -> None:
        """Mark the current provider response as complete."""
        self._response_idle.set()

    async def aclose(self) -> None:
        """Cancel queued work and close the scheduler."""
        self._closed = True
        async with self._condition:
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)

    async def _run(self) -> None:
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: self._items or self._closed)
                if self._closed:
                    return
                priority, sequence, before_response = heapq.heappop(self._items)
            await self._response_idle.wait()
            await self._user_idle.wait()
            await self._transport.wait_output_idle()
            await self._user_idle.wait()
            async with self._condition:
                if self._items and self._items[0][0] < priority:
                    heapq.heappush(self._items, (priority, sequence, before_response))
                    continue
            if before_response is not None:
                should_respond = await before_response()
                if should_respond is False:
                    continue
            await self._user_idle.wait()
            async with self._condition:
                if self._items and self._items[0][0] < priority:
                    continue
            self._response_idle.clear()
            try:
                await self._connection.response.create()
            except Exception:
                self._response_idle.set()
                raise


async def truncate_playback(connection: Any, transport: VoiceTransport) -> str | None:
    """Clear unheard audio and align Realtime context to the heard endpoint."""
    receipt = await transport.interrupt_output()
    if receipt is None or receipt.stream_id is None:
        return None
    await connection.conversation.item.truncate(
        item_id=receipt.stream_id,
        content_index=receipt.content_index,
        audio_end_ms=receipt.audio_end_ms,
    )
    return receipt.stream_id


def _runtime_context_item(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": text}],
    }


@asynccontextmanager
async def _maybe_trace(
    raw: Any,
    *,
    thread_id: str,
    project_name: str | None,
    model: str,
    transport: VoiceTransport,
) -> AsyncIterator[Any]:
    if not os.getenv("LANGSMITH_API_KEY"):
        yield raw
        return
    try:
        from langsmith.integrations.openai_realtime import (  # type: ignore[import-not-found]  # noqa: PLC0415 - optional
            wrap_realtime,
        )
    except ImportError:
        yield raw
        return
    async with wrap_realtime(
        raw,
        thread_id=thread_id,
        sample_rate=SAMPLE_RATE,
        project_name=project_name,
        tags=["langchain-voice", "openai"],
        metadata={"realtime_model": model},
        is_agent_speaking=lambda: transport.output_active,
    ) as connection:
        yield connection


def _record_audio(connection: Any, method: str, pcm: bytes) -> None:
    recorder = getattr(connection, method, None)
    if callable(recorder):
        recorder(pcm)


async def _run(
    session: VoiceSession,
    *,
    transport: VoiceTransport,
    ui: StatusUI | None,
    project_name: str | None,
    model: str | None,
    voice: str | None,
) -> None:
    """Run the complete LangChain Voice orchestration loop over OpenAI Realtime."""
    if not os.getenv("OPENAI_API_KEY"):
        msg = "OPENAI_API_KEY is not set"
        raise RuntimeError(msg)
    try:
        from openai import AsyncOpenAI  # noqa: PLC0415 - optional provider dependency
    except ImportError as exc:  # pragma: no cover - environment dependent
        msg = "OpenAI support requires `pip install langchain[voice-openai]`"
        raise RuntimeError(msg) from exc

    ui = ui or NullUI()
    model = model or DEFAULT_MODEL
    voice = voice or DEFAULT_VOICE
    client = AsyncOpenAI()
    thread_id = str(uuid.uuid4())
    interrupted_items: set[str] = set()
    assistant_transcripts: dict[str, str] = {}
    finalizers: set[asyncio.Task[None]] = set()
    mic_task: asyncio.Task[None] | None = None
    event_task: asyncio.Task[None] | None = None
    scheduler: ResponseScheduler | None = None
    pending_results: list[str] = []
    pending_results_lock = asyncio.Lock()

    ui.log(f"[langchain.voice] connecting to OpenAI Realtime with model={model}...")
    try:
        async with (
            client.realtime.connect(model=model) as raw,
            _maybe_trace(
                raw,
                thread_id=thread_id,
                project_name=project_name,
                model=model,
                transport=transport,
            ) as connection,
        ):
            await connection.session.update(
                session=session_config(session.instructions, voice=voice)
            )

            scheduler = ResponseScheduler(connection, transport)
            scheduler.start()

            async def inject_pending_results() -> bool:
                async with pending_results_lock:
                    if not pending_results:
                        return False
                    results = list(pending_results)
                    pending_results.clear()
                for text in results:
                    await connection.conversation.item.create(item=_runtime_context_item(text))
                return True

            async def prepare_foreground_response() -> bool:
                await inject_pending_results()
                return True

            async def deliver_task_result(event: TerminalTaskEvent) -> None:
                text = format_task_event(event)
                async with pending_results_lock:
                    pending_results.append(text)
                await scheduler.request(BACKGROUND_RESPONSE, inject_pending_results)

            event_task = asyncio.create_task(
                relay_task_results(session, deliver_task_result),
                name="langchain-voice-events",
            )
            ui.log("[langchain.voice] connected. Talk into your mic — Ctrl-C to quit.")
            ui.set_state("listening")

            async def pump_transport() -> None:
                async for media_event in transport.events():
                    if isinstance(media_event, TransportDisconnected):
                        msg = media_event.reason or "voice transport disconnected"
                        raise ConnectionError(msg)
                    if isinstance(media_event, AudioPlayed):
                        _record_audio(connection, "record_agent_audio", media_event.frame.data)
                        continue
                    if not isinstance(media_event, AudioReceived):
                        continue
                    frame = media_event.frame
                    if frame.format is not AudioFormat.PCM_S16LE or frame.channels != 1:
                        msg = "OpenAI Realtime requires mono PCM16 input audio"
                        raise ValueError(msg)
                    sent = resample_pcm16(frame.data, frame.sample_rate, SAMPLE_RATE)
                    await connection.input_audio_buffer.append(
                        audio=base64.b64encode(sent).decode("ascii")
                    )
                    _record_audio(connection, "record_user_audio", sent)
                    ui.update_level(frame_level(frame.data))

            mic_task = asyncio.create_task(pump_transport(), name="langchain-voice-transport")

            async def record_assistant_when_heard(item_id: str) -> None:
                await transport.wait_output_idle()
                transcript = assistant_transcripts.pop(item_id, "").strip()
                was_interrupted = item_id in interrupted_items
                interrupted_items.discard(item_id)
                if transcript and not was_interrupted:
                    await session.record_transcript("assistant", transcript)

            async for event in connection:
                event_type = event.type
                if event_type == "response.output_audio.delta":
                    await transport.send_audio(
                        AudioFrame(
                            data=base64.b64decode(event.delta),
                            sample_rate=SAMPLE_RATE,
                        ),
                        stream_id=event.item_id,
                        content_index=getattr(event, "content_index", 0),
                    )
                    ui.set_state("speaking")
                elif event_type == "input_audio_buffer.speech_started":
                    scheduler.user_speech_started()
                    was_speaking = not scheduler.response_idle or transport.output_active
                    if was_speaking:
                        item_id = await truncate_playback(connection, transport)
                        if item_id is not None:
                            interrupted_items.add(item_id)
                    else:
                        await transport.interrupt_output()
                    ui.set_state("hearing you")
                elif event_type == "input_audio_buffer.speech_stopped":
                    ui.set_state("transcribing")
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = (event.transcript or "").strip()
                    if transcript:
                        ui.log(f"user:  {transcript}")
                        await session.record_transcript("user", transcript)
                        ui.set_state("thinking")
                        await scheduler.request(USER_RESPONSE, prepare_foreground_response)
                        scheduler.user_speech_finished()
                    else:
                        scheduler.user_speech_finished()
                        ui.set_state("listening")
                elif event_type == "conversation.item.input_audio_transcription.failed":
                    ui.log("[langchain.voice] transcription failed")
                    scheduler.user_speech_finished()
                    ui.set_state("listening")
                elif event_type == "response.output_audio_transcript.done":
                    transcript = (event.transcript or "").strip()
                    if transcript:
                        assistant_transcripts[event.item_id] = transcript
                        ui.log(f"agent: {transcript}")
                elif event_type == "response.done":
                    calls = [
                        item
                        for item in (event.response.output or [])
                        if item.type == "function_call"
                    ]
                    if calls:
                        ui.set_state("coordinating")
                        results = await asyncio.gather(
                            *(
                                execute_task_tool(session, call.name, call.arguments)
                                for call in calls
                            )
                        )
                        for call, result in zip(calls, results, strict=True):
                            await connection.conversation.item.create(
                                item={
                                    "type": "function_call_output",
                                    "call_id": call.call_id,
                                    "output": json.dumps(
                                        result,
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    ),
                                }
                            )
                        await scheduler.request(TOOL_RESPONSE, prepare_foreground_response)
                    else:
                        for item in event.response.output or []:
                            item_id = getattr(item, "id", None)
                            if item_id in assistant_transcripts:
                                task = asyncio.create_task(record_assistant_when_heard(item_id))
                                finalizers.add(task)
                                task.add_done_callback(finalizers.discard)
                        ui.set_state("listening")
                    scheduler.mark_done()
                elif event_type == "error":
                    ui.log(f"[langchain.voice] provider error: {event.error}")
                    scheduler.mark_done()
    finally:
        for pending_task in (mic_task, event_task, *finalizers):
            if pending_task is not None:
                pending_task.cancel()
        await asyncio.gather(
            *(pending_task for pending_task in (mic_task, event_task, *finalizers) if pending_task),
            return_exceptions=True,
        )
        if scheduler is not None:
            await scheduler.aclose()
        ui.finish()
