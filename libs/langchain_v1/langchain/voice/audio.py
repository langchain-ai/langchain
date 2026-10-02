"""Legacy local-audio adapters and status contracts for LangChain Voice."""

from __future__ import annotations

import array
import math
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


@runtime_checkable
class AudioInput(Protocol):
    """A legacy source of PCM16 mono audio frames.

    New media integrations should implement `VoiceTransport` instead.
    """

    sample_rate: int

    def start(self) -> None:
        """Start capturing audio."""
        ...

    def frames(self) -> AsyncIterator[bytes]:
        """Yield captured PCM16 frames."""
        ...

    def stop(self) -> None:
        """Stop capturing audio and release resources."""
        ...


@runtime_checkable
class AudioOutput(Protocol):
    """A legacy PCM16 mono audio sink with interruptible playback.

    `LocalAudioTransport` adapts this device-shaped interface to the provider-neutral
    transport contract.
    """

    sample_rate: int

    def start(self) -> None:
        """Start audio playback."""
        ...

    def write(self, data: bytes) -> None:
        """Queue a PCM16 frame for playback."""
        ...

    def buffered_bytes(self) -> int:
        """Return the number of queued audio bytes."""
        ...

    def clear(self) -> None:
        """Discard audio that has not yet played."""
        ...

    def set_played_callback(self, callback: Callable[[bytes], None] | None) -> None:
        """Set a callback invoked with audio after it has played."""
        ...

    def stop(self) -> None:
        """Stop playback and release resources."""
        ...


@runtime_checkable
class StatusUI(Protocol):
    """Optional observer for frontend status and transcript updates."""

    def set_state(self, state: str) -> None:
        """Publish a human-readable runtime state."""
        ...

    def update_level(self, level: float) -> None:
        """Publish the current microphone level."""
        ...

    def log(self, message: str) -> None:
        """Publish a transcript or diagnostic message."""
        ...

    def finish(self) -> None:
        """Mark the session as finished."""
        ...


class NullUI:
    """No-op status observer used when an application does not supply one."""

    def set_state(self, state: str) -> None:
        """Ignore a state update."""
        del state

    def update_level(self, level: float) -> None:
        """Ignore a microphone-level update."""
        del level

    def log(self, message: str) -> None:
        """Ignore a log message."""
        del message

    def finish(self) -> None:
        """Finish without side effects."""
        return


def frame_level(frame: bytes) -> float:
    """Map a PCM16 mono frame to an approximate 0..1 input level."""
    samples = array.array("h")
    samples.frombytes(frame)
    if not samples:
        return 0.0
    mean_sq = sum(sample * sample for sample in samples) / len(samples)
    if mean_sq <= 0:
        return 0.0
    db = 20.0 * math.log10(math.sqrt(mean_sq) / 32767.0)
    return max(0.0, min(1.0, (db + 60.0) / 60.0))


def resample_pcm16(data: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linearly resample little-endian PCM16 mono without extra dependencies."""
    if not data or source_rate == target_rate:
        return data
    if source_rate <= 0 or target_rate <= 0:
        msg = "sample rates must be positive"
        raise ValueError(msg)
    source = array.array("h")
    source.frombytes(data)
    if not source:
        return b""
    target_size = max(1, round(len(source) * target_rate / source_rate))
    if target_size == 1 or len(source) == 1:
        return array.array("h", [source[0]] * target_size).tobytes()
    scale = (len(source) - 1) / (target_size - 1)
    target = array.array("h")
    for index in range(target_size):
        position = index * scale
        left = int(position)
        right = min(left + 1, len(source) - 1)
        fraction = position - left
        target.append(round(source[left] * (1 - fraction) + source[right] * fraction))
    return target.tobytes()
