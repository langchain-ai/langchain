from __future__ import annotations

import asyncio
import unittest
from typing import TYPE_CHECKING

import pytest

from langchain.voice.transport import (
    AudioFormat,
    AudioFrame,
    AudioIOTransport,
    AudioPlayed,
    AudioReceived,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


class FakeAudioInput:
    sample_rate = 16_000

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.release = asyncio.Event()

    def start(self) -> None:
        self.started += 1

    async def frames(self) -> AsyncIterator[bytes]:
        yield b"\x01\x00" * 160
        await self.release.wait()

    def stop(self) -> None:
        self.stopped += 1
        self.release.set()


class FakeAudioOutput:
    sample_rate = 24_000

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.buffer = bytearray()
        self.callback: Callable[[bytes], None] | None = None
        self.clear_count = 0

    def start(self) -> None:
        self.started += 1

    def write(self, data: bytes) -> None:
        self.buffer.extend(data)

    def buffered_bytes(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.clear_count += 1
        self.buffer.clear()

    def set_played_callback(self, callback: Callable[[bytes], None] | None) -> None:
        self.callback = callback

    def stop(self) -> None:
        self.stopped += 1

    def play(self, byte_count: int) -> None:
        data = bytes(self.buffer[:byte_count])
        del self.buffer[:byte_count]
        if self.callback is not None:
            self.callback(data)


class TestAudioFrame:
    def test_frame_is_self_describing_and_counts_samples(self) -> None:
        frame = AudioFrame(data=b"\x00\x00" * 80, sample_rate=16_000)

        assert frame.format is AudioFormat.PCM_S16LE
        assert frame.channels == 1
        assert frame.sample_count == 80

    def test_rejects_incomplete_pcm_samples(self) -> None:
        with pytest.raises(ValueError, match="complete samples"):
            AudioFrame(data=b"\x00", sample_rate=16_000)


class AudioIOTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapts_media_and_reports_the_heard_endpoint(self) -> None:
        audio_in = FakeAudioInput()
        audio_out = FakeAudioOutput()
        transport = AudioIOTransport(audio_in, audio_out)
        events = transport.events()

        await transport.start()
        received = await anext(events)
        assert isinstance(received, AudioReceived)
        assert received.frame.sample_rate == 16_000

        output = AudioFrame(data=b"\x02\x00" * 2_400, sample_rate=24_000)
        await transport.send_audio(output, stream_id="item-1", content_index=2)
        audio_out.play(2_400)
        played = await anext(events)
        assert isinstance(played, AudioPlayed)
        assert played.stream_id == "item-1"
        assert played.content_index == 2

        receipt = await transport.interrupt_output()
        assert receipt is not None
        assert receipt.stream_id == "item-1"
        assert receipt.content_index == 2
        assert receipt.played_samples == 1_200
        assert receipt.audio_end_ms == 50
        assert audio_out.clear_count == 1

        await transport.aclose()
        assert audio_in.started == audio_out.started == 1
        assert audio_in.stopped == audio_out.stopped == 1
        assert audio_out.callback is None

    async def test_rejects_output_with_an_incompatible_sample_rate(self) -> None:
        transport = AudioIOTransport(FakeAudioInput(), FakeAudioOutput())
        await transport.start()
        self.addAsyncCleanup(transport.aclose)

        with pytest.raises(ValueError, match="sample rate"):
            await transport.send_audio(AudioFrame(data=b"\x00\x00", sample_rate=16_000))
