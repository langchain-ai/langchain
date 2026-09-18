from __future__ import annotations

import asyncio
import sys
import unittest
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from langchain.voice import AudioFrame
from langchain.voice.transport import AudioPlayed, AudioReceived
from langchain.voice.transports.livekit import LiveKitAudioTransport

if TYPE_CHECKING:
    from typing import ClassVar


class FakeAudioFrame:
    def __init__(
        self,
        *,
        data: bytes,
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int,
    ) -> None:
        self.data = memoryview(data)
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel


class FakeAudioFrameEvent:
    def __init__(self, frame: FakeAudioFrame) -> None:
        self.frame = frame


class FakeAudioStream:
    instances: ClassVar[list[FakeAudioStream]] = []
    _closed = object()

    def __init__(self, track: Any, *, sample_rate: int, num_channels: int) -> None:
        self.track = track
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False
        self.instances.append(self)

    def __aiter__(self) -> FakeAudioStream:
        return self

    async def __anext__(self) -> Any:
        item = await self.queue.get()
        if item is self._closed:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        self.closed = True
        self.queue.put_nowait(self._closed)


class FakeAudioSource:
    def __init__(
        self,
        sample_rate: int,
        num_channels: int,
        *,
        queue_size_ms: int,
    ) -> None:
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.queue_size_ms = queue_size_ms
        self.queued_duration = 0.0
        self.frames: list[FakeAudioFrame] = []
        self.clear_count = 0
        self.closed = False

    async def capture_frame(self, frame: FakeAudioFrame) -> None:
        self.frames.append(frame)
        self.queued_duration += frame.samples_per_channel / frame.sample_rate

    async def wait_for_playout(self) -> None:
        self.queued_duration = 0.0

    def clear_queue(self) -> None:
        self.clear_count += 1
        self.queued_duration = 0.0

    async def aclose(self) -> None:
        self.closed = True


class FakeLocalAudioTrack:
    def __init__(self, name: str, source: FakeAudioSource) -> None:
        self.name = name
        self.source = source
        self.closed = False

    @classmethod
    def create_audio_track(cls, name: str, source: FakeAudioSource) -> FakeLocalAudioTrack:
        return cls(name, source)

    async def aclose(self) -> None:
        self.closed = True


class FakeLocalParticipant:
    def __init__(self) -> None:
        self.published: list[tuple[Any, Any]] = []
        self.unpublished: list[str] = []

    async def publish_track(self, track: Any, options: Any) -> Any:
        self.published.append((track, options))
        return SimpleNamespace(sid="published-track")

    async def unpublish_track(self, sid: str) -> None:
        self.unpublished.append(sid)


class FakeRoom:
    def __init__(self) -> None:
        self.local_participant = FakeLocalParticipant()
        self.remote_participants: dict[str, Any] = {}
        self.handlers: dict[str, list[Any]] = {}

    def on(self, event: str) -> Any:
        def register(handler: Any) -> Any:
            self.handlers.setdefault(event, []).append(handler)
            return handler

        return register

    def off(self, event: str, handler: Any) -> None:
        self.handlers[event].remove(handler)

    def emit(self, event: str, *args: Any) -> None:
        for handler in self.handlers.get(event, []):
            handler(*args)


class FakeTrack:
    kind = "audio"

    def __init__(self, sid: str) -> None:
        self.sid = sid


def fake_livekit_module() -> ModuleType:
    rtc = SimpleNamespace(
        AudioFrame=FakeAudioFrame,
        AudioFrameEvent=FakeAudioFrameEvent,
        AudioSource=FakeAudioSource,
        AudioStream=FakeAudioStream,
        LocalAudioTrack=FakeLocalAudioTrack,
        TrackKind=SimpleNamespace(KIND_AUDIO="audio"),
        TrackPublishOptions=type("TrackPublishOptions", (), {}),
        TrackSource=SimpleNamespace(SOURCE_MICROPHONE="microphone"),
    )
    module = ModuleType("livekit")
    module.rtc = rtc  # type: ignore[attr-defined]
    return module


class LiveKitAudioTransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        FakeAudioStream.instances.clear()

    async def test_filters_participants_and_converts_input_audio(self) -> None:
        room = FakeRoom()
        transport = LiveKitAudioTransport(room, participant_identity="customer")
        events = transport.events()

        with patch.dict(sys.modules, {"livekit": fake_livekit_module()}):
            await transport.start()
            room.emit(
                "track_subscribed",
                FakeTrack("ignored"),
                object(),
                SimpleNamespace(identity="someone-else"),
            )
            room.emit(
                "track_subscribed",
                FakeTrack("customer-track"),
                object(),
                SimpleNamespace(identity="customer"),
            )

        assert len(FakeAudioStream.instances) == 1
        stream = FakeAudioStream.instances[0]
        frame = FakeAudioFrame(
            data=b"\x01\x00" * 240,
            sample_rate=24_000,
            num_channels=1,
            samples_per_channel=240,
        )
        stream.queue.put_nowait(FakeAudioFrameEvent(frame))

        received = await asyncio.wait_for(anext(events), 1)
        assert isinstance(received, AudioReceived)
        assert received.frame.data == bytes(frame.data)
        assert received.frame.sample_rate == 24_000
        await transport.aclose()

    async def test_interruption_reports_only_audio_that_played(self) -> None:
        room = FakeRoom()
        transport = LiveKitAudioTransport(room)

        with patch.dict(sys.modules, {"livekit": fake_livekit_module()}):
            await transport.start()

        await transport.send_audio(
            AudioFrame(b"\x02\x00" * 2_400, 24_000),
            stream_id="item-1",
            content_index=3,
        )
        transport._audio_source.queued_duration = 0.05

        receipt = await transport.interrupt_output()
        played = await anext(transport.events())

        assert receipt is not None
        assert receipt.stream_id == "item-1"
        assert receipt.content_index == 3
        assert receipt.played_samples == 1_200
        assert receipt.audio_end_ms == 50
        assert isinstance(played, AudioPlayed)
        assert played.frame.sample_count == 1_200
        await transport.aclose()

    async def test_cleanup_unpublishes_only_adapter_owned_track(self) -> None:
        room = FakeRoom()
        transport = LiveKitAudioTransport(room)

        with patch.dict(sys.modules, {"livekit": fake_livekit_module()}):
            await transport.start()
        await transport.aclose()

        assert room.local_participant.unpublished == ["published-track"]
        assert all(not handlers for handlers in room.handlers.values())
        assert transport._audio_source.closed


if __name__ == "__main__":
    unittest.main()
