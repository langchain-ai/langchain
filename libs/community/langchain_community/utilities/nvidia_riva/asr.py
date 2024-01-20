"""An NVIDIA Riva ASR module that converts audio bytes into LLM ready text."""
import asyncio
import logging
import queue
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
    cast,
)

from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.runnables import RunnableConfig, RunnableSerializable

from .common import RivaAuthMixin, RivaCommonConfigMixin, SentinelT, _import_riva_client

if TYPE_CHECKING:
    import riva.client
    import riva.client.proto.riva_asr_pb2 as rasr

_LOGGER = logging.getLogger(__name__)
_QUEUE_GET_TIMEOUT = 0.5
HANGUP = SentinelT()


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


StreamInputType = Union[bytes, SentinelT]
StreamOutputType = str


class AudioStream:
    """A message containing streaming audio."""

    _put_lock: threading.Lock
    _queue: queue.Queue[StreamInputType]
    output: queue.Queue[StreamOutputType]
    hangup: _Event
    user_talking: _Event
    user_quiet: _Event
    _worker: Optional[threading.Thread]

    def __init__(self, maxsize: int = 0) -> None:
        """Initialize the queue."""
        self._put_lock = threading.Lock()
        self._queue = queue.Queue(maxsize=maxsize)
        self.output = queue.Queue()
        self.hangup = _Event()
        self.user_quiet = _Event()
        self.user_talking = _Event()
        self._worker = None

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

    async def __aiter__(self) -> AsyncIterator[StreamInputType]:
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
        input_done = self.hungup and self.empty
        output_done = (
            self._worker is not None
            and not self._worker.is_alive()
            and self.output.empty()
        )
        return input_done and output_done

    @property
    def running(self) -> bool:
        """Indicate if the ASR stream is running."""
        if self._worker:
            return self._worker.is_alive()
        return False

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

    def register(self, responses: Iterator["rasr.StreamingRecognizeResponse"]) -> None:
        """Drain the responses from the provided iterator and put them into a queue."""
        if self.running:
            raise RuntimeError("An ASR instance has already been registered.")

        has_started = threading.Barrier(2, timeout=5)

        def worker() -> None:
            """Consume the ASR Generator."""
            has_started.wait()
            for response in responses:
                if not response.results:
                    continue

                for result in response.results:
                    if not result.alternatives:
                        continue

                    if result.is_final:
                        self.user_talking.clear()
                        self.user_quiet.set()
                        transcript = cast(str, result.alternatives[0].transcript)
                        self.output.put(transcript)

                    elif not self.user_talking.is_set():
                        self.user_talking.set()
                        self.user_quiet.clear()

        self._worker = threading.Thread(target=worker)
        self._worker.daemon = True
        self._worker.start()
        has_started.wait()


ASRInputType = AudioStream
ASROutputType = str


class RivaASR(
    RivaAuthMixin,
    RivaCommonConfigMixin,
    RunnableSerializable[ASRInputType, ASROutputType],
):
    """A runnable that performs Automatic Speech Recognition (ASR) using NVIDIA Riva."""

    name: str = "nvidia_riva_asr"
    description: str = (
        "A Runnable for converting audio bytes to a string."
        "This is useful for feeding an audio stream into a chain and"
        "preprocessing that audio to create an LLM prompt."
    )

    # riva options
    audio_channel_count: int = Field(
        1, description="The number of audio channels in the input audio stream."
    )
    profanity_filter: bool = Field(
        True,
        description="Controls whether or not Riva should attempt to filter profanity out of the transcribed text.",
    )
    enable_automatic_punctuation: bool = Field(
        True,
        description="Controls whether Riva should attempt to correct senetence puncuation in the transcribed text.",
    )

    @root_validator(pre=True)
    @classmethod
    def _validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the Python environment and input arguments."""
        _ = _import_riva_client()
        return values

    @property
    def config(self) -> "riva.client.StreamingRecognitionConfig":
        """Create and return the riva config object."""
        riva_client = _import_riva_client()
        return riva_client.StreamingRecognitionConfig(
            interim_results=True,
            config=riva_client.RecognitionConfig(
                encoding=self.encoding,
                sample_rate_hertz=self.sample_rate_hertz,
                audio_channel_count=self.audio_channel_count,
                max_alternatives=1,
                profanity_filter=self.profanity_filter,
                enable_automatic_punctuation=self.enable_automatic_punctuation,
                language_code=self.language_code,
            ),
        )

    def _get_service(self) -> "riva.client.ASRService":
        """Connect to the riva service and return the a client object."""
        riva_client = _import_riva_client()
        try:
            return riva_client.ASRService(self.auth)
        except Exception as err:
            raise ValueError(
                "Error raised while connecting to the Riva ASR server."
            ) from err

    def invoke(
        self,
        input: ASRInputType,
        _: Optional[RunnableConfig] = None,
        **__: Optional[Any],
    ) -> ASROutputType:
        """Transcribe the audio bytes into a string with Riva stop when the person stops talking."""
        # create an output text generator with Riva
        if not input.running:
            service = self._get_service()
            responses = service.streaming_response_generator(
                audio_chunks=input,
                streaming_config=self.config,
            )
            input.register(responses)

        # return the first valid result
        full_response: List[str] = []
        while not input.complete:
            with input.output.not_empty:
                ready = input.output.not_empty.wait(0.1)

            if ready:
                while not input.output.empty():
                    try:
                        full_response += [input.output.get_nowait()]
                    except queue.Empty:
                        continue
                    input.output.task_done()
                _LOGGER.debug("Riva ASR returning: %s", repr(full_response))
                return " ".join(full_response).strip()

        return ""
