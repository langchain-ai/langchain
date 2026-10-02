"""Gemini Live conversation runtime owned entirely by LangChain Voice."""

from __future__ import annotations

import asyncio
import os
import uuid
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
    relay_task_results,
    task_event_payload,
    task_tool_schema,
)
from langchain.voice.diagnostics import debug_refs as _debug_refs
from langchain.voice.diagnostics import make_debug_log as _make_debug_log
from langchain.voice.transport import (
    AudioFormat,
    AudioFrame,
    AudioPlayed,
    AudioReceived,
    TransportDisconnected,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from langchain.voice.agent import VoiceSession
    from langchain.voice.transport import VoiceTransport

DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_VOICE = "Aoede"
SEND_SAMPLE_RATE = 16_000
RECEIVE_SAMPLE_RATE = 24_000
MAX_TRANSCRIPT_CHARS = 4_000


@dataclass(frozen=True, slots=True)
class GeminiLiveConversationLayer:
    """Configured Gemini Live conversation runtime."""

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
        """Run one Gemini Live conversation using an isolated voice session."""
        await _run(
            session,
            transport=transport,
            ui=ui,
            project_name=project_name,
            model=self.model,
            voice=self.voice,
        )


def _append_transcript(current: str, fragment: str) -> str:
    remaining = MAX_TRANSCRIPT_CHARS - len(current)
    return current if remaining <= 0 else current + fragment[:remaining]


class LiveMessage:
    """Provider-independent view of application-relevant Gemini messages."""

    def __init__(self, raw: Any) -> None:
        """Wrap one raw Gemini Live server message."""
        self.raw = raw

    @property
    def server_content(self) -> Any:
        """Return the message's server content, if present."""
        return getattr(self.raw, "server_content", None)

    @property
    def interrupted(self) -> bool:
        """Return whether Gemini interrupted the current model turn."""
        content = self.server_content
        return bool(content is not None and getattr(content, "interrupted", False))

    @property
    def turn_complete(self) -> bool:
        """Return whether Gemini completed the current model turn."""
        content = self.server_content
        return bool(content is not None and getattr(content, "turn_complete", False))

    def _transcription(self, name: str) -> Any:
        content = self.server_content
        return getattr(content, name, None) if content is not None else None

    def _transcript(self, name: str) -> str | None:
        transcription = self._transcription(name)
        text = getattr(transcription, "text", None) if transcription is not None else None
        return str(text) if text else None

    def _finished(self, name: str) -> bool:
        transcription = self._transcription(name)
        return bool(transcription is not None and getattr(transcription, "finished", False))

    @property
    def user_transcript(self) -> str | None:
        """Return the latest user transcript fragment."""
        return self._transcript("input_transcription")

    @property
    def user_transcript_finished(self) -> bool:
        """Return whether the current user transcript is final."""
        return self._finished("input_transcription")

    @property
    def assistant_transcript(self) -> str | None:
        """Return the latest assistant transcript fragment."""
        return self._transcript("output_transcription")

    @property
    def assistant_transcript_finished(self) -> bool:
        """Return whether the current assistant transcript is final."""
        return self._finished("output_transcription")

    @property
    def audio_chunks(self) -> list[bytes]:
        """Return PCM audio chunks carried by this message."""
        content = self.server_content
        turn = getattr(content, "model_turn", None) if content is not None else None
        parts = getattr(turn, "parts", None) if turn is not None else None
        chunks: list[bytes] = []
        for part in parts or []:
            blob = getattr(part, "inline_data", None)
            data = getattr(blob, "data", None) if blob is not None else None
            if isinstance(data, bytes) and data:
                chunks.append(data)
        return chunks

    @property
    def function_calls(self) -> list[Any]:
        """Return task coordination calls carried by this message."""
        tool_call = getattr(self.raw, "tool_call", None)
        return list(getattr(tool_call, "function_calls", None) or [])


def gemini_task_tool(types: Any) -> Any:
    """Build Gemini declarations for LangChain Voice's fixed task contract."""
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=spec.name,
                description=spec.description,
                parameters_json_schema=task_tool_schema(spec),
                behavior=(
                    types.Behavior.NON_BLOCKING
                    if spec.name == "create_task"
                    else types.Behavior.BLOCKING
                ),
            )
            for spec in TASK_TOOL_SPECS
        ]
    )


def _function_response(
    types: Any,
    call: Any,
    response: dict[str, Any],
    *,
    will_continue: bool,
    scheduling: Any,
) -> Any:
    """Build a Gemini response that participates in async task delivery."""
    return types.FunctionResponse(
        id=call.id,
        name=call.name,
        response=response,
        will_continue=will_continue,
        scheduling=scheduling,
    )


async def _deliver_terminal_task_results(
    connection: Any,
    types: Any,
    results: list[tuple[Any, TerminalTaskEvent]],
) -> None:
    """Deliver a result batch with one generation, then close it silently."""
    if not results:
        return

    await connection.send_tool_response(
        function_responses=[
            _function_response(
                types,
                call,
                {"output": task_event_payload(event)},
                will_continue=True,
                scheduling=(
                    types.FunctionResponseScheduling.WHEN_IDLE
                    if index == len(results) - 1
                    else types.FunctionResponseScheduling.SILENT
                ),
            )
            for index, (call, event) in enumerate(results)
        ]
    )
    await connection.send_tool_response(
        function_responses=[
            _function_response(
                types,
                call,
                {},
                will_continue=False,
                scheduling=types.FunctionResponseScheduling.SILENT,
            )
            for call, _event in results
        ]
    )


class _TerminalTaskResultDispatcher:
    """Serialize result-driven turns and coalesce results waiting for idle."""

    def __init__(
        self,
        connection: Any,
        types: Any,
        debug_log: Callable[[str], None] | None = None,
    ) -> None:
        self._connection = connection
        self._types = types
        self._debug = debug_log or (lambda _message: None)
        self._pending: asyncio.Queue[tuple[Any, TerminalTaskEvent]] = asyncio.Queue()
        self._conversation_idle = asyncio.Event()
        self._conversation_idle.set()
        self._batch_sequence = 0
        self._inflight_task_ids: tuple[str, ...] = ()

    @property
    def pending_count(self) -> int:
        return self._pending.qsize()

    @property
    def inflight_task_ids(self) -> tuple[str, ...]:
        return self._inflight_task_ids

    def mark_busy(self, reason: str = "runtime") -> None:
        if self._conversation_idle.is_set():
            self._debug(f"conversation_busy reason={reason}")
        self._conversation_idle.clear()

    def mark_idle(self, reason: str = "runtime") -> None:
        if not self._conversation_idle.is_set():
            self._debug(f"conversation_idle reason={reason} pending={self.pending_count}")
        if self._inflight_task_ids:
            self._debug(
                f"result_followup_complete task_ids={_debug_refs(list(self._inflight_task_ids))}"
            )
            self._inflight_task_ids = ()
        self._conversation_idle.set()

    async def enqueue(self, call: Any, event: TerminalTaskEvent) -> None:
        await self._pending.put((call, event))
        self._debug(
            f"result_queued event={event.type} "
            f"task_id={_debug_refs([event.task_id])} "
            f"call_id={_debug_refs([getattr(call, 'id', 'unknown')])} "
            f"pending={self.pending_count}"
        )

    async def run(self) -> None:
        while True:
            first = await self._pending.get()
            await self._conversation_idle.wait()

            # Let terminal events already ready in the event loop join this turn.
            await asyncio.sleep(0)
            batch = [first]
            while True:
                try:
                    batch.append(self._pending.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # A later result must wait for this generated follow-up to complete.
            self._batch_sequence += 1
            self._inflight_task_ids = tuple(event.task_id for _call, event in batch)
            self._debug(
                f"result_batch_dispatch sequence={self._batch_sequence} "
                f"count={len(batch)} task_ids="
                f"{_debug_refs(list(self._inflight_task_ids))}"
            )
            self.mark_busy(reason="result_batch_dispatch")
            await _deliver_terminal_task_results(
                self._connection,
                self._types,
                batch,
            )
            self._debug(
                f"result_batch_sent sequence={self._batch_sequence} "
                f"generation_triggers=1 silent_closures={len(batch)}"
            )


def live_config(types: Any, instructions: str, *, voice: str = DEFAULT_VOICE) -> Any:
    """Build a Gemini Live configuration with LangChain Voice task tools."""
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=instructions,
        tools=[gemini_task_tool(types)],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                start_of_speech_sensitivity=(types.StartSensitivity.START_SENSITIVITY_HIGH),
                end_of_speech_sensitivity=(types.EndSensitivity.END_SENSITIVITY_LOW),
                prefix_padding_ms=200,
                silence_duration_ms=800,
            )
        ),
    )


@asynccontextmanager
async def _maybe_trace(
    raw: Any,
    *,
    model: str,
    thread_id: str,
    project_name: str | None,
    transport: VoiceTransport,
) -> AsyncIterator[Any]:
    if not os.getenv("LANGSMITH_API_KEY"):
        yield raw
        return
    try:
        from langsmith.integrations.gemini_live import (  # type: ignore[import-not-found]  # noqa: PLC0415 - optional
            wrap_gemini_live,
        )
    except ImportError:
        yield raw
        return
    async with wrap_gemini_live(
        raw,
        model=model,
        thread_id=thread_id,
        sample_rate=RECEIVE_SAMPLE_RATE,
        project_name=project_name,
        tags=["langchain-voice", "gemini"],
        metadata={"live_model": model},
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
    """Run the complete LangChain Voice orchestration loop over Gemini Live."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        msg = "GOOGLE_API_KEY is not set"
        raise RuntimeError(msg)
    try:
        from google import genai  # noqa: PLC0415 - optional provider dependency
        from google.genai import types  # noqa: PLC0415 - optional provider dependency
    except ImportError as exc:  # pragma: no cover - environment dependent
        msg = "Gemini support requires `pip install langchain[voice-gemini]`"
        raise RuntimeError(msg) from exc

    ui = ui or NullUI()
    debug_log = _make_debug_log(ui)
    model = model or DEFAULT_MODEL
    voice = voice or DEFAULT_VOICE
    client = genai.Client(api_key=api_key)
    thread_id = str(uuid.uuid4())
    task_calls: dict[str, Any] = {}
    task_calls_changed = asyncio.Condition()
    mic_task: asyncio.Task[None] | None = None
    event_task: asyncio.Task[None] | None = None
    response_task: asyncio.Task[None] | None = None
    result_dispatch_task: asyncio.Task[None] | None = None
    transcript_tasks: set[asyncio.Task[None]] = set()

    ui.log(f"[langchain.voice] connecting to Gemini Live with model={model}...")
    try:
        async with (
            client.aio.live.connect(
                model=model,
                config=live_config(types, session.instructions, voice=voice),
            ) as raw,
            _maybe_trace(
                raw,
                model=model,
                thread_id=thread_id,
                project_name=project_name,
                transport=transport,
            ) as connection,
        ):
            result_dispatcher = _TerminalTaskResultDispatcher(connection, types, debug_log)
            debug_log(
                "session_connected provider=gemini "
                f"model={_debug_refs([model])} "
                f"thread_id={_debug_refs([thread_id])}"
            )
            ui.log("[langchain.voice] connected. Talk into your mic — Ctrl-C to quit.")
            ui.set_state("listening")

            async def pump_transport() -> None:
                async for event in transport.events():
                    if isinstance(event, TransportDisconnected):
                        msg = event.reason or "voice transport disconnected"
                        raise ConnectionError(msg)
                    if isinstance(event, AudioPlayed):
                        _record_audio(connection, "record_agent_audio", event.frame.data)
                        continue
                    if not isinstance(event, AudioReceived):
                        continue
                    frame = event.frame
                    if frame.format is not AudioFormat.PCM_S16LE or frame.channels != 1:
                        msg = "Gemini Live requires mono PCM16 input audio"
                        raise ValueError(msg)
                    sent = resample_pcm16(frame.data, frame.sample_rate, SEND_SAMPLE_RATE)
                    await connection.send_realtime_input(
                        audio=types.Blob(
                            data=sent,
                            mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}",
                        )
                    )
                    _record_audio(
                        connection,
                        "record_user_audio",
                        resample_pcm16(frame.data, frame.sample_rate, RECEIVE_SAMPLE_RATE),
                    )
                    ui.update_level(frame_level(frame.data))

            async def deliver_task_result(event: TerminalTaskEvent) -> None:
                debug_log(
                    f"terminal_event_observed event={event.type} "
                    f"task_id={_debug_refs([event.task_id])} "
                    f"revision={event.revision}"
                )
                async with task_calls_changed:
                    await task_calls_changed.wait_for(lambda: event.task_id in task_calls)
                    call = task_calls.pop(event.task_id)
                debug_log(
                    "terminal_event_matched "
                    f"task_id={_debug_refs([event.task_id])} "
                    f"call_id={_debug_refs([getattr(call, 'id', 'unknown')])}"
                )
                await result_dispatcher.enqueue(call, event)

            async def record_assistant(text: str) -> None:
                await transport.wait_output_idle()
                if text.strip():
                    await session.record_transcript("assistant", text)

            async def pump_responses() -> None:
                user_transcript = ""
                assistant_transcript = ""
                turn_sequence = 0
                while True:
                    async for raw_message in connection.receive():
                        message = LiveMessage(raw_message)
                        chunks = message.audio_chunks
                        if message.function_calls or (
                            message.server_content is not None and not message.turn_complete
                        ):
                            result_dispatcher.mark_busy(reason="provider_activity")
                        if chunks or message.function_calls or message.assistant_transcript:
                            ui.set_state("speaking" if chunks else "coordinating")
                        if message.interrupted:
                            await transport.interrupt_output()
                            assistant_transcript = ""
                            ui.set_state("hearing you")
                        for chunk in chunks:
                            await transport.send_audio(
                                AudioFrame(data=chunk, sample_rate=RECEIVE_SAMPLE_RATE)
                            )
                            ui.set_state("speaking")
                        if fragment := message.user_transcript:
                            user_transcript = _append_transcript(user_transcript, fragment)
                            ui.set_state("hearing you")
                        if message.user_transcript_finished and user_transcript:
                            ui.log(f"user:  {user_transcript}")
                            await session.record_transcript("user", user_transcript)
                            user_transcript = ""
                        if fragment := message.assistant_transcript:
                            assistant_transcript = _append_transcript(
                                assistant_transcript, fragment
                            )
                        if message.function_calls:
                            ui.set_state("coordinating")
                            valid_calls = [
                                call for call in message.function_calls if getattr(call, "id", None)
                            ]
                            debug_log(
                                f"tool_calls_received count={len(valid_calls)} "
                                "calls="
                                + _debug_refs(
                                    [
                                        f"{getattr(call, 'name', 'unknown')}:"
                                        f"{getattr(call, 'id', 'unknown')}"
                                        for call in valid_calls
                                    ]
                                )
                            )
                            results = await asyncio.gather(
                                *(
                                    execute_task_tool(
                                        session,
                                        getattr(call, "name", ""),
                                        getattr(call, "args", None),
                                    )
                                    for call in valid_calls
                                )
                            )
                            responses = []
                            for index, (call, result) in enumerate(
                                zip(valid_calls, results, strict=True)
                            ):
                                continues = bool(
                                    call.name == "create_task"
                                    and result.get("ok")
                                    and result.get("task_id")
                                )
                                if continues:
                                    async with task_calls_changed:
                                        task_calls[result["task_id"]] = call
                                        task_calls_changed.notify_all()
                                responses.append(
                                    _function_response(
                                        types,
                                        call,
                                        result,
                                        will_continue=continues,
                                        scheduling=(
                                            types.FunctionResponseScheduling.WHEN_IDLE
                                            if index == len(results) - 1
                                            else types.FunctionResponseScheduling.SILENT
                                        ),
                                    )
                                )
                            if responses:
                                await connection.send_tool_response(function_responses=responses)
                                debug_log(
                                    "tool_responses_sent "
                                    f"count={len(responses)} "
                                    "continuing_task_ids="
                                    + _debug_refs(
                                        [
                                            result.get("task_id")
                                            for result in results
                                            if result.get("status") == "started"
                                        ]
                                    )
                                    + " generation_triggers=1"
                                )
                        if message.turn_complete:
                            turn_sequence += 1
                            debug_log(
                                f"provider_turn_complete turn={turn_sequence} "
                                f"assistant_chars={len(assistant_transcript)} "
                                f"pending_results={result_dispatcher.pending_count} "
                                "inflight_task_ids="
                                + _debug_refs(list(result_dispatcher.inflight_task_ids))
                            )
                            if user_transcript:
                                ui.log(f"user:  {user_transcript}")
                                await session.record_transcript("user", user_transcript)
                                user_transcript = ""
                            if assistant_transcript:
                                text = assistant_transcript
                                ui.log(f"agent: {text}")
                                assistant_transcript = ""
                                task = asyncio.create_task(record_assistant(text))
                                transcript_tasks.add(task)
                                task.add_done_callback(transcript_tasks.discard)
                            result_dispatcher.mark_idle(reason="provider_turn_complete")
                            ui.set_state("listening")

            mic_task = asyncio.create_task(pump_transport(), name="langchain-voice-transport")
            event_task = asyncio.create_task(
                relay_task_results(session, deliver_task_result),
                name="langchain-voice-events",
            )
            response_task = asyncio.create_task(pump_responses(), name="langchain-voice-responses")
            result_dispatch_task = asyncio.create_task(
                result_dispatcher.run(), name="langchain-voice-result-dispatch"
            )
            await asyncio.gather(
                mic_task,
                event_task,
                response_task,
                result_dispatch_task,
            )
    finally:
        for task in (
            mic_task,
            event_task,
            response_task,
            result_dispatch_task,
            *transcript_tasks,
        ):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(
                task
                for task in (
                    mic_task,
                    event_task,
                    response_task,
                    result_dispatch_task,
                    *transcript_tasks,
                )
                if task
            ),
            return_exceptions=True,
        )
        ui.finish()
