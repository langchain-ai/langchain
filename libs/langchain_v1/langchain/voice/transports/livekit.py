"""LiveKit room audio transport for LangChain Voice."""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain.voice.transport import (
    AudioFormat,
    AudioFrame,
    AudioPlayed,
    AudioReceived,
    PlaybackReceipt,
    TransportDisconnected,
    TransportEvent,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable


@dataclass(slots=True)
class _PendingAudio:
    frame: AudioFrame
    stream_id: str | None
    content_index: int
    offset_samples: int = 0

    @property
    def remaining_samples(self) -> int:
        return self.frame.sample_count - self.offset_samples


_EVENTS_CLOSED = object()


class LiveKitAudioTransport:
    """Exchange audio with one participant in a caller-owned LiveKit room.

    The room must already be connected. This transport publishes its own output
    track and subscribes to one remote participant's audio, but it never connects
    or disconnects the room itself.
    """

    def __init__(
        self,
        room: Any,
        *,
        participant_identity: str | None = None,
        input_sample_rate: int = 24_000,
        output_sample_rate: int = 24_000,
        queue_size_ms: int = 1_000,
        track_name: str = "langchain-voice",
    ) -> None:
        """Initialize a transport for an existing LiveKit room.

        Args:
            room: An already-connected `livekit.rtc.Room`.
            participant_identity: Exact remote identity to consume. When omitted,
                the first remote participant with an audio track is selected.
            input_sample_rate: Sample rate requested from LiveKit input streams.
            output_sample_rate: Sample rate accepted by `send_audio`.
            queue_size_ms: Maximum native LiveKit output queue duration.
            track_name: Name of the published assistant audio track.
        """
        if input_sample_rate <= 0 or output_sample_rate <= 0:
            msg = "LiveKit audio sample rates must be positive"
            raise ValueError(msg)
        if queue_size_ms <= 0:
            msg = "queue_size_ms must be positive"
            raise ValueError(msg)
        if not track_name.strip():
            msg = "track_name must be non-empty"
            raise ValueError(msg)
        self._room = room
        self._participant_identity = participant_identity
        self._selected_identity: str | None = None
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._queue_size_ms = queue_size_ms
        self._track_name = track_name
        self._events: asyncio.Queue[TransportEvent | object] = asyncio.Queue()
        self._rtc: Any = None
        self._audio_source: Any = None
        self._audio_track: Any = None
        self._publication: Any = None
        self._handlers: list[tuple[str, Any]] = []
        self._input_streams: dict[str, tuple[Any, asyncio.Task[None]]] = {}
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._pending: deque[_PendingAudio] = deque()
        self._active_stream_id: str | None = None
        self._active_content_index = 0
        self._active_played_samples = 0
        self._started = False
        self._closed = False

    @property
    def output_sample_rate(self) -> int:
        """Return the sample rate accepted for assistant audio."""
        return self._output_sample_rate

    @property
    def output_active(self) -> bool:
        """Return whether LiveKit still has assistant audio queued."""
        if self._audio_source is None:
            return False
        return float(getattr(self._audio_source, "queued_duration", 0.0)) > 0

    async def start(self) -> None:
        """Publish output audio and begin consuming the selected participant."""
        if self._closed:
            msg = "the LiveKit audio transport is closed"
            raise RuntimeError(msg)
        if self._started:
            return
        try:
            from livekit import rtc  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            msg = "LiveKit audio requires `pip install livekit`"
            raise RuntimeError(msg) from exc

        self._rtc = rtc
        self._register_room_handler("track_subscribed", self._on_track_subscribed)
        self._register_room_handler("track_unsubscribed", self._on_track_unsubscribed)
        self._register_room_handler("participant_disconnected", self._on_participant_disconnected)
        self._register_room_handler("disconnected", self._on_room_disconnected)

        try:
            self._audio_source = rtc.AudioSource(
                self._output_sample_rate,
                1,
                queue_size_ms=self._queue_size_ms,
            )
            self._audio_track = rtc.LocalAudioTrack.create_audio_track(
                self._track_name, self._audio_source
            )
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            self._publication = await self._room.local_participant.publish_track(
                self._audio_track, options
            )
            self._started = True
            self._subscribe_existing_audio_tracks()
        except Exception:
            await self.aclose()
            raise

    async def events(self) -> AsyncIterator[TransportEvent]:
        """Yield room audio, playback acknowledgements, and disconnects."""
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
        """Publish one mono PCM16 frame to the LiveKit room."""
        self._ensure_started()
        if frame.format is not AudioFormat.PCM_S16LE or frame.channels != 1:
            msg = "LiveKitAudioTransport requires mono PCM16 output"
            raise ValueError(msg)
        if frame.sample_rate != self.output_sample_rate:
            msg = (
                "output frame sample rate does not match the transport: "
                f"{frame.sample_rate} != {self.output_sample_rate}"
            )
            raise ValueError(msg)
        if not frame.data:
            return

        key = (stream_id, content_index)
        if key != (self._active_stream_id, self._active_content_index):
            self._active_stream_id = stream_id
            self._active_content_index = content_index
            self._active_played_samples = 0
        pending = _PendingAudio(frame, stream_id, content_index)
        self._pending.append(pending)
        rtc_frame = self._rtc.AudioFrame(
            data=frame.data,
            sample_rate=frame.sample_rate,
            num_channels=frame.channels,
            samples_per_channel=frame.sample_count,
        )
        try:
            await self._audio_source.capture_frame(rtc_frame)
        except Exception:
            self._pending.remove(pending)
            raise

    async def wait_output_idle(self) -> None:
        """Wait for LiveKit playout, then acknowledge all pending audio."""
        self._ensure_started()
        await self._audio_source.wait_for_playout()
        self._acknowledge_played(sum(item.remaining_samples for item in self._pending))

    async def interrupt_output(self) -> PlaybackReceipt | None:
        """Clear LiveKit's queue and return the assistant audio heard so far."""
        self._ensure_started()
        total_pending = sum(item.remaining_samples for item in self._pending)
        queued_duration = max(0.0, float(getattr(self._audio_source, "queued_duration", 0.0)))
        queued_samples = min(
            total_pending,
            round(queued_duration * self.output_sample_rate),
        )
        self._acknowledge_played(total_pending - queued_samples)
        self._audio_source.clear_queue()
        self._pending.clear()

        stream_id = self._active_stream_id
        content_index = self._active_content_index
        played_samples = self._active_played_samples
        self._active_stream_id = None
        self._active_content_index = 0
        self._active_played_samples = 0
        if stream_id is None:
            return None
        return PlaybackReceipt(
            stream_id=stream_id,
            content_index=content_index,
            played_samples=played_samples,
            sample_rate=self.output_sample_rate,
        )

    async def aclose(self) -> None:
        """Release adapter-owned streams and tracks without closing the room."""
        if self._closed:
            return
        self._closed = True
        self._unregister_room_handlers()
        await self._close_input_streams()
        await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
        if self._audio_source is not None:
            self._audio_source.clear_queue()
        self._pending.clear()
        await self._unpublish_output_track()
        await self._maybe_aclose(self._audio_source)
        self._events.put_nowait(_EVENTS_CLOSED)

    def _register_room_handler(self, event: str, handler: Any) -> None:
        self._room.on(event)(handler)
        self._handlers.append((event, handler))

    def _unregister_room_handlers(self) -> None:
        off = getattr(self._room, "off", None)
        if callable(off):
            for event, handler in self._handlers:
                off(event, handler)
        self._handlers.clear()

    def _subscribe_existing_audio_tracks(self) -> None:
        for participant in self._room.remote_participants.values():
            for publication in participant.track_publications.values():
                track = getattr(publication, "track", None)
                if track is not None:
                    self._on_track_subscribed(track, publication, participant)

    def _on_track_subscribed(self, track: Any, publication: Any, participant: Any) -> None:
        del publication
        if track.kind != self._rtc.TrackKind.KIND_AUDIO:
            return
        identity = str(participant.identity)
        if self._participant_identity is not None and identity != self._participant_identity:
            return
        if self._selected_identity is None:
            self._selected_identity = identity
        if identity != self._selected_identity:
            return
        track_id = str(track.sid)
        if track_id in self._input_streams:
            return
        stream = self._rtc.AudioStream(
            track,
            sample_rate=self._input_sample_rate,
            num_channels=1,
        )
        task = asyncio.create_task(
            self._pump_audio_stream(stream),
            name=f"langchain-voice-livekit-input-{track_id}",
        )
        self._input_streams[track_id] = (stream, task)

    def _on_track_unsubscribed(self, track: Any, publication: Any, participant: Any) -> None:
        del publication, participant
        entry = self._input_streams.pop(str(track.sid), None)
        if entry is not None:
            task = asyncio.create_task(self._close_input_stream(*entry))
            self._cleanup_tasks.add(task)
            task.add_done_callback(self._cleanup_tasks.discard)

    def _on_participant_disconnected(self, participant: Any) -> None:
        if str(participant.identity) != self._selected_identity:
            return
        self._events.put_nowait(
            TransportDisconnected(reason="selected LiveKit participant disconnected")
        )

    def _on_room_disconnected(self, *args: Any) -> None:
        del args
        self._events.put_nowait(TransportDisconnected(reason="LiveKit room disconnected"))

    async def _pump_audio_stream(self, stream: Any) -> None:
        try:
            async for event in stream:
                frame = event.frame
                await self._events.put(
                    AudioReceived(
                        AudioFrame(
                            data=frame.data.tobytes(),
                            sample_rate=int(frame.sample_rate),
                            channels=int(frame.num_channels),
                        )
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._events.put(TransportDisconnected(reason=str(exc)))
            raise

    def _acknowledge_played(self, sample_count: int) -> None:
        remaining = sample_count
        while remaining > 0 and self._pending:
            pending = self._pending[0]
            played = min(remaining, pending.remaining_samples)
            start = pending.offset_samples * 2
            end = start + played * 2
            chunk = pending.frame.data[start:end]
            pending.offset_samples += played
            remaining -= played
            if (pending.stream_id, pending.content_index) == (
                self._active_stream_id,
                self._active_content_index,
            ):
                self._active_played_samples += played
            self._events.put_nowait(
                AudioPlayed(
                    AudioFrame(chunk, pending.frame.sample_rate),
                    stream_id=pending.stream_id,
                    content_index=pending.content_index,
                )
            )
            if pending.remaining_samples == 0:
                self._pending.popleft()

    async def _close_input_streams(self) -> None:
        entries = list(self._input_streams.values())
        self._input_streams.clear()
        await asyncio.gather(
            *(self._close_input_stream(stream, task) for stream, task in entries),
            return_exceptions=True,
        )

    async def _close_input_stream(self, stream: Any, task: asyncio.Task[None]) -> None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await self._maybe_aclose(stream)

    async def _unpublish_output_track(self) -> None:
        if self._publication is None:
            return
        track_sid = getattr(self._publication, "sid", None)
        unpublish = getattr(self._room.local_participant, "unpublish_track", None)
        if track_sid is not None and callable(unpublish):
            await unpublish(track_sid)
        await self._maybe_aclose(self._audio_track)

    @staticmethod
    async def _maybe_aclose(resource: Any) -> None:
        close = getattr(resource, "aclose", None)
        if not callable(close):
            return
        result: Awaitable[Any] | Any = close()
        if inspect.isawaitable(result):
            await result

    def _ensure_started(self) -> None:
        if not self._started or self._closed:
            msg = "the LiveKit audio transport is not running"
            raise RuntimeError(msg)
