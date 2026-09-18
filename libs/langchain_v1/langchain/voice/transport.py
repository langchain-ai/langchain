"""Provider-neutral media transport contracts for LangChain Voice."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain.voice.audio import AudioInput, AudioOutput


class AudioFormat(str, Enum):
    """Audio sample encodings understood by voice transports."""

    PCM_S16LE = "pcm_s16le"


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """One self-describing frame of audio media."""

    data: bytes
    sample_rate: int
    channels: int = 1
    format: AudioFormat = AudioFormat.PCM_S16LE

    def __post_init__(self) -> None:
        """Validate frame metadata and sample alignment."""
        if self.sample_rate <= 0:
            msg = "audio frame sample_rate must be positive"
            raise ValueError(msg)
        if self.channels <= 0:
            msg = "audio frame channels must be positive"
            raise ValueError(msg)
        if self.format is AudioFormat.PCM_S16LE and len(self.data) % (2 * self.channels):
            msg = "PCM16 audio must contain complete samples for every channel"
            raise ValueError(msg)

    @property
    def sample_count(self) -> int:
        """Return the number of samples per channel in this frame."""
        return len(self.data) // (2 * self.channels)


@dataclass(frozen=True, slots=True)
class AudioReceived:
    """Audio captured from the user and ready for a conversation provider."""

    frame: AudioFrame


@dataclass(frozen=True, slots=True)
class AudioPlayed:
    """Audio confirmed as played to the user by the transport."""

    frame: AudioFrame
    stream_id: str | None = None
    content_index: int = 0


@dataclass(frozen=True, slots=True)
class UserSpeechStarted:
    """Transport-level signal that the user started speaking."""


@dataclass(frozen=True, slots=True)
class UserSpeechEnded:
    """Transport-level signal that the user stopped speaking."""


@dataclass(frozen=True, slots=True)
class TransportDisconnected:
    """Signal that the remote or local media source disconnected."""

    reason: str | None = None


TransportEvent = (
    AudioReceived | AudioPlayed | UserSpeechStarted | UserSpeechEnded | TransportDisconnected
)


@dataclass(frozen=True, slots=True)
class PlaybackReceipt:
    """The heard endpoint of one interrupted output stream."""

    stream_id: str | None
    content_index: int
    played_samples: int
    sample_rate: int

    @property
    def audio_end_ms(self) -> int:
        """Return the played duration in whole milliseconds."""
        return self.played_samples * 1000 // self.sample_rate


@runtime_checkable
class VoiceTransport(Protocol):
    """Duplex media boundary between a voice session and its application."""

    @property
    def output_sample_rate(self) -> int:
        """Return the sample rate expected for output audio."""
        ...

    @property
    def output_active(self) -> bool:
        """Return whether output audio is still queued or playing."""
        ...

    async def start(self) -> None:
        """Open transport resources and begin receiving media."""
        ...

    def events(self) -> AsyncIterator[TransportEvent]:
        """Yield input, playback, and transport-control events."""
        ...

    async def send_audio(
        self,
        frame: AudioFrame,
        *,
        stream_id: str | None = None,
        content_index: int = 0,
    ) -> None:
        """Queue one output frame for playback."""
        ...

    async def wait_output_idle(self) -> None:
        """Wait until all queued output has played or been discarded."""
        ...

    async def interrupt_output(self) -> PlaybackReceipt | None:
        """Discard unheard output and return the last heard endpoint."""
        ...

    async def aclose(self) -> None:
        """Close transport resources and stop media delivery."""
        ...


@dataclass(slots=True)
class _QueuedAudio:
    stream_id: str | None
    content_index: int
    remaining_bytes: int


_EVENTS_CLOSED = object()


class LocalAudioTransport:
    """Adapt legacy local-style audio input and output to `VoiceTransport`.

    This adapter keeps device-specific callbacks and queue inspection out of
    conversation providers. New network and room transports should implement
    `VoiceTransport` directly.
    """

    def __init__(self, audio_in: AudioInput, audio_out: AudioOutput) -> None:
        """Initialize an adapter around an existing mic and speaker pair.

        Args:
            audio_in: Source of mono PCM16 audio.
            audio_out: Interruptible sink for mono PCM16 audio.
        """
        self._audio_in = audio_in
        self._audio_out = audio_out
        self._events: asyncio.Queue[TransportEvent | object] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._input_task: asyncio.Task[None] | None = None
        self._started = False
        self._closed = False
        self._lock = threading.Lock()
        self._queued: deque[_QueuedAudio] = deque()
        self._active_stream_id: str | None = None
        self._active_content_index = 0
        self._active_played_bytes = 0

    @property
    def output_sample_rate(self) -> int:
        """Return the legacy output sink's sample rate."""
        return self._audio_out.sample_rate

    @property
    def output_active(self) -> bool:
        """Return whether the legacy sink still has queued bytes."""
        return self._audio_out.buffered_bytes() > 0

    async def start(self) -> None:
        """Start the wrapped input and output exactly once."""
        if self._closed:
            msg = "the voice transport is closed"
            raise RuntimeError(msg)
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        try:
            self._audio_in.start()
            self._audio_out.start()
            self._audio_out.set_played_callback(self._on_played)
        except Exception:
            self._audio_out.set_played_callback(None)
            self._audio_in.stop()
            self._audio_out.stop()
            raise
        self._started = True
        self._input_task = asyncio.create_task(
            self._pump_input(), name="langchain-voice-transport-input"
        )

    async def events(self) -> AsyncIterator[TransportEvent]:
        """Yield typed media events until the transport closes."""
        while True:
            item = await self._events.get()
            if item is _EVENTS_CLOSED:
                return
            if isinstance(item, (AudioReceived, AudioPlayed, TransportDisconnected)):
                yield item

    async def send_audio(
        self,
        frame: AudioFrame,
        *,
        stream_id: str | None = None,
        content_index: int = 0,
    ) -> None:
        """Queue a mono PCM16 frame on the wrapped output sink."""
        self._ensure_started()
        if frame.format is not AudioFormat.PCM_S16LE or frame.channels != 1:
            msg = "LocalAudioTransport requires mono PCM16 output"
            raise ValueError(msg)
        if frame.sample_rate != self.output_sample_rate:
            msg = (
                "output frame sample rate does not match the transport: "
                f"{frame.sample_rate} != {self.output_sample_rate}"
            )
            raise ValueError(msg)
        if not frame.data:
            return
        queued = _QueuedAudio(stream_id, content_index, len(frame.data))
        with self._lock:
            self._queued.append(queued)
            if (stream_id, content_index) != (
                self._active_stream_id,
                self._active_content_index,
            ):
                self._active_stream_id = stream_id
                self._active_content_index = content_index
                self._active_played_bytes = 0
        try:
            self._audio_out.write(frame.data)
        except Exception:
            with self._lock, suppress(ValueError):
                self._queued.remove(queued)
            raise

    async def wait_output_idle(self) -> None:
        """Wait for the wrapped output sink to drain."""
        self._ensure_started()
        while self._audio_out.buffered_bytes() > 0:  # noqa: ASYNC110
            await asyncio.sleep(0.01)

    async def interrupt_output(self) -> PlaybackReceipt | None:
        """Clear unheard audio and return what the sink confirmed as played."""
        self._ensure_started()
        self._audio_out.clear()
        with self._lock:
            self._queued.clear()
            stream_id = self._active_stream_id
            content_index = self._active_content_index
            played_bytes = self._active_played_bytes
            self._active_stream_id = None
            self._active_content_index = 0
            self._active_played_bytes = 0
        if stream_id is None:
            return None
        return PlaybackReceipt(
            stream_id=stream_id,
            content_index=content_index,
            played_samples=played_bytes // 2,
            sample_rate=self.output_sample_rate,
        )

    async def aclose(self) -> None:
        """Stop wrapped resources and close the event stream."""
        if self._closed:
            return
        self._closed = True
        if self._input_task is not None:
            self._input_task.cancel()
            await asyncio.gather(self._input_task, return_exceptions=True)
        self._audio_out.set_played_callback(None)
        if self._started:
            self._audio_in.stop()
            self._audio_out.stop()
        with self._lock:
            self._queued.clear()
        self._events.put_nowait(_EVENTS_CLOSED)

    async def _pump_input(self) -> None:
        try:
            async for data in self._audio_in.frames():
                await self._events.put(
                    AudioReceived(AudioFrame(data=data, sample_rate=self._audio_in.sample_rate))
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._events.put(TransportDisconnected(reason=str(exc)))
            raise

    def _on_played(self, data: bytes) -> None:
        loop = self._loop
        if loop is None or self._closed:
            return
        events: list[AudioPlayed] = []
        offset = 0
        with self._lock:
            while offset < len(data) and self._queued:
                queued = self._queued[0]
                length = min(len(data) - offset, queued.remaining_bytes)
                chunk = data[offset : offset + length]
                offset += length
                queued.remaining_bytes -= length
                if (queued.stream_id, queued.content_index) == (
                    self._active_stream_id,
                    self._active_content_index,
                ):
                    self._active_played_bytes += length
                events.append(
                    AudioPlayed(
                        frame=AudioFrame(data=chunk, sample_rate=self.output_sample_rate),
                        stream_id=queued.stream_id,
                        content_index=queued.content_index,
                    )
                )
                if queued.remaining_bytes == 0:
                    self._queued.popleft()
        for event in events:
            loop.call_soon_threadsafe(self._events.put_nowait, event)

    def _ensure_started(self) -> None:
        if not self._started or self._closed:
            msg = "the voice transport is not running"
            raise RuntimeError(msg)


# Compatibility name used by the initial experimental release.
AudioIOTransport = LocalAudioTransport
