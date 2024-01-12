"""An NVIDIA Riva TTS module that converts text into audio bytes."""
import asyncio
import queue
import threading
from typing import TYPE_CHECKING, AsyncIterator, Iterator, Optional, Union

from .common import SentinelT

if TYPE_CHECKING:
    import riva.client.proto.riva_asr_pb2 as rasr

_QUEUE_GET_TIMEOUT = 0.5
HANGUP = SentinelT()
StreamInputType = Union[bytes, SentinelT]
StreamOutputType = bytes


class _Event:
    """A combined event that is threadsafe and async safe."""

    _event: threading.Event
    _aevent: asyncio.Event

    def __init__(self) -> None:
        """Initialize the event."""
        self._event = threading.Event()
        self._aevent = asyncio.Event()

    def set(self) -> None:
        """Set the event."""
        self._event.set()
        self._aevent.set()

    def clear(self) -> None:
        """Set the event."""
        self._event.clear()
        self._aevent.clear()

    def is_set(self) -> bool:
        """Indicate if the event is set."""
        return self._event.is_set()

    def wait(self) -> None:
        """Wait for the event to be set."""
        self._event.wait()

    async def async_wait(self) -> None:
        """Async wait for the event to be set."""
        await self._aevent.wait()


class AudioStream:
    """A message containing streaming audio."""

    _put_lock: threading.Lock
    _queue: queue.Queue[StreamInputType]
    output: Optional[Iterator["rasr.StreamingRecognizeResponse"]]
    output_complete: bool
    hangup: _Event
    transcript_complete: _Event

    def __init__(self, maxsize: int = 0) -> None:
        """Initialize the queue."""
        self._put_lock = threading.Lock()
        self._queue = queue.Queue(maxsize=maxsize)
        self.output = None
        self.output_complete = False
        self.hangup = _Event()
        self.transcript_complete = _Event()

    def __iter__(self) -> None:
        """Return an error."""
        while True:
            # get next item
            try:
                next_val = self._queue.get(True, _QUEUE_GET_TIMEOUT)
            except queue.Empty:
                continue

            # hangup when requested
            if next_val == HANGUP:
                break

            # yield next item
            yield next_val
            self._queue.task_done()

    async def __aiter__(self) -> AsyncIterator[StreamOutputType]:
        """Iterate through all items in the queue until the Hangup Sentinel is observed."""
        while True:
            # get next item
            try:
                next_val = await asyncio.get_event_loop().run_in_executor(
                    None, self._queue.get, True, _QUEUE_GET_TIMEOUT
                )
            except queue.Empty:
                continue

            # hangup when requested
            if next_val == HANGUP:
                break

            # yield next item
            yield next_val
            self._queue.task_done()

    @property
    def hungup(self) -> bool:
        """Indicate if the audio stream has hungup."""
        return self.hangup.is_set()

    @property
    def empty(self) -> bool:
        """Indicate in the input stream buffer is empty."""
        return self._queue.empty()

    @property
    def complete(self) -> bool:
        """Indicate if the audio stream has hungup and been processed."""
        return self.hungup and self.empty and self.transcript_complete.is_set()

    def put(self, item: StreamInputType, timeout: Optional[int] = None) -> None:
        """Put a new item into the queue."""
        with self._put_lock:
            if self.hungup:
                raise RuntimeError(
                    "The audio stream has already been hungup. Cannot put more data."
                )
            if item is HANGUP:
                self.hangup.set()
            self._queue.put(item, timeout=timeout)

    async def aput(self, item: StreamInputType, timeout: Optional[int] = None) -> None:
        """Async put a new item into the queue."""
        loop = asyncio.get_event_loop()
        await asyncio.wait_for(loop.run_in_executor(None, self.put, item), timeout)

    def close(self, timeout: Optional[int] = None) -> None:
        """Send the hangup signal."""
        self.put(HANGUP, timeout)

    async def aclose(self, timeout: Optional[int] = None) -> None:
        """Async send the hangup signal."""
        await self.aput(HANGUP, timeout)
