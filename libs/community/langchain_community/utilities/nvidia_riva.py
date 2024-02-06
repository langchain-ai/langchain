"""A common module for NVIDIA Riva Runnables."""
import asyncio
import logging
import pathlib
import queue
import tempfile
import threading
import wave
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import (
    AnyHttpUrl,
    BaseModel,
    Field,
    parse_obj_as,
    root_validator,
    validator,
)
from langchain_core.runnables import RunnableConfig, RunnableSerializable

if TYPE_CHECKING:
    import riva.client
    import riva.client.proto.riva_asr_pb2 as rasr

_LOGGER = logging.getLogger(__name__)
_QUEUE_GET_TIMEOUT = 0.5
_MAX_TEXT_LENGTH = 400
_SENTENCE_TERMINATORS = ("\n", ".", "!", "?", "¡", "¿")


# COMMON utilities used by all Riva Runnables
def _import_riva_client() -> "riva.client":
    """Import the riva client and raise an error on failure."""
    try:
        # pylint: disable-next=import-outside-toplevel # this client library is optional
        import riva.client
    except ImportError as err:
        raise ImportError(
            "Could not import the NVIDIA Riva client library. "
            "Please install it with `pip install nvidia-riva-client`."
        ) from err
    return riva.client


class SentinelT:  # pylint: disable=too-few-public-methods
    """An empty Sentinel type."""


HANGUP = SentinelT()
_TRANSFORM_END = SentinelT()


class RivaAudioEncoding(str, Enum):
    """An enum of the possible choices for Riva audio encoding.

    The list of types exposed by the Riva GRPC Protobuf files can be found
    with the following commands:
    ```python
    import riva.client
    print(riva.client.AudioEncoding.keys())
    ```
    """

    ALAW = "ALAW"
    ENCODING_UNSPECIFIED = "ENCODING_UNSPECIFIED"
    FLAC = "FLAC"
    LINEAR_PCM = "LINEAR_PCM"
    MULAW = "MULAW"
    OGGOPUS = "OGGOPUS"

    @classmethod
    def from_wave_format_code(cls, format_code: int) -> "RivaAudioEncoding":
        """Return the audio encoding specified by the format code in the wave file.

        ref: https://mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
        """
        try:
            return {1: cls.LINEAR_PCM, 6: cls.ALAW, 7: cls.MULAW}[format_code]
        except KeyError as err:
            raise NotImplementedError(
                "The following wave file format code is "
                f"not supported by Riva: {format_code}"
            ) from err

    @property
    def riva_pb2(self) -> "riva.client.AudioEncoding":
        """Returns the Riva API object for the encoding."""
        riva_client = _import_riva_client()
        return getattr(riva_client.AudioEncoding, self)


class RivaAuthMixin(BaseModel):
    """Configuration for the authentication to a Riva service connection."""

    url: Union[AnyHttpUrl, str] = Field(
        AnyHttpUrl("http://localhost:50051", scheme="http"),
        description="The full URL where the Riva service can be found.",
        examples=["http://localhost:50051", "https://user@pass:riva.example.com"],
    )
    ssl_cert: Optional[str] = Field(
        None,
        description="A full path to the file where Riva's public ssl key can be read.",
    )

    @property
    def auth(self) -> "riva.client.Auth":
        """Return a riva client auth object."""
        riva_client = _import_riva_client()
        url = cast(AnyHttpUrl, self.url)
        use_ssl = url.scheme == "https"  # pylint: disable=no-member # false positive
        url_no_scheme = str(self.url).split("/")[2]
        return riva_client.Auth(
            ssl_cert=self.ssl_cert, use_ssl=use_ssl, uri=url_no_scheme
        )

    @validator("url", pre=True, allow_reuse=True)
    @classmethod
    def _validate_url(cls, val: Any) -> AnyHttpUrl:
        """Do some initial conversations for the URL before checking."""
        if isinstance(val, str):
            return cast(AnyHttpUrl, parse_obj_as(AnyHttpUrl, val))
        return cast(AnyHttpUrl, val)


class RivaCommonConfigMixin(BaseModel):
    """A collection of common Riva settings."""

    encoding: RivaAudioEncoding = Field(
        default=RivaAudioEncoding.LINEAR_PCM,
        description="The encoding on the audio stream.",
    )
    sample_rate_hertz: int = Field(
        default=8000, description="The sample rate frequency of audio stream."
    )
    language_code: str = Field(
        default="en-US",
        description=(
            "The [BCP-47 language code]"
            "(https://www.rfc-editor.org/rfc/bcp/bcp47.txt) for "
            "the target language."
        ),
    )


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


def _mk_wave_file(
    output_directory: Optional[str], sample_rate: float
) -> Tuple[Optional[str], Optional[wave.Wave_write]]:
    """Create a new wave file and return the wave write object and filename."""
    if output_directory:
        with tempfile.NamedTemporaryFile(
            mode="bx", suffix=".wav", delete=False, dir=output_directory
        ) as f:
            wav_file_name = f.name
        wav_file = wave.open(wav_file_name, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        return (wav_file_name, wav_file)
    return (None, None)


def _coerce_string(val: "TTSInputType") -> str:
    """Attempt to coerce the input value to a string.

    This is particularly useful for converting LangChain message to strings.
    """
    if isinstance(val, PromptValue):
        return val.to_string()
    if isinstance(val, BaseMessage):
        return str(val.content)
    return str(val)


def _process_chunks(inputs: Iterator["TTSInputType"]) -> Generator[str, None, None]:
    """Filter the input chunks are return strings ready for TTS."""
    buffer = ""
    for chunk in inputs:
        chunk = _coerce_string(chunk)

        # return the buffer if an end of sentence character is detected
        for terminator in _SENTENCE_TERMINATORS:
            while terminator in chunk:
                last_sentence, chunk = chunk.split(terminator, 1)
                yield buffer + last_sentence + terminator
                buffer = ""

        buffer += chunk

        # return the buffer if is too long
        if len(buffer) > _MAX_TEXT_LENGTH:
            for idx in range(0, len(buffer), _MAX_TEXT_LENGTH):
                yield buffer[idx : idx + 5]
            buffer = ""

    # return remaining buffer
    if buffer:
        yield buffer


# Riva AudioStream Type
StreamInputType = Union[bytes, SentinelT]
StreamOutputType = str


class AudioStream:
    """A message containing streaming audio."""

    _put_lock: threading.Lock
    _queue: queue.Queue
    output: queue.Queue
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

    def __iter__(self) -> Generator[bytes, None, None]:
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
        """Iterate through all items in the queue until HANGUP."""
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


# RivaASR Runnable
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
        description=(
            "Controls whether or not Riva should attempt to filter "
            "profanity out of the transcribed text."
        ),
    )
    enable_automatic_punctuation: bool = Field(
        True,
        description=(
            "Controls whether Riva should attempt to correct "
            "senetence puncuation in the transcribed text."
        ),
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
    ) -> ASROutputType:
        """Transcribe the audio bytes into a string with Riva."""
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


# RivaTTS Runnable
# pylint: disable-next=invalid-name
TTSInputType = Union[str, AnyMessage, PromptValue]
TTSOutputType = bytes


class RivaTTS(
    RivaAuthMixin,
    RivaCommonConfigMixin,
    RunnableSerializable[TTSInputType, TTSOutputType],
):
    """A runnable that performs Text-to-Speech (TTS) with NVIDIA Riva."""

    name: str = "nvidia_riva_tts"
    description: str = (
        "A tool for converting text to speech."
        "This is useful for converting LLM output into audio bytes."
    )

    # riva options
    voice_name: str = Field(
        "English-US.Female-1",
        description=(
            "The voice model in Riva to use for speech. "
            "Pre-trained models are documented in "
            "[the Riva documentation]"
            "(https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tts/tts-overview.html)."
        ),
    )
    output_directory: Optional[str] = Field(
        None,
        description=(
            "The directory where all audio files should be saved. "
            "A null value indicates that wave files should not be saved. "
            "This is useful for debugging purposes."
        ),
    )

    @root_validator(pre=True)
    @classmethod
    def _validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the Python environment and input arguments."""
        _ = _import_riva_client()
        return values

    @validator("output_directory")
    @classmethod
    def _output_directory_validator(cls, v: str) -> str:
        if v:
            dirpath = pathlib.Path(v)
            dirpath.mkdir(parents=True, exist_ok=True)
            return str(dirpath.absolute())
        return v

    def _get_service(self) -> "riva.client.SpeechSynthesisService":
        """Connect to the riva service and return the a client object."""
        riva_client = _import_riva_client()
        try:
            return riva_client.SpeechSynthesisService(self.auth)
        except Exception as err:
            raise ValueError(
                "Error raised while connecting to the Riva TTS server."
            ) from err

    def invoke(
        self, input: TTSInputType, _: Union[RunnableConfig, None] = None
    ) -> TTSOutputType:
        """Perform TTS by taking a string and outputting the entire audio file."""
        return b"".join(self.transform(iter([input])))

    def transform(
        self,
        input: Iterator[TTSInputType],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[TTSOutputType]:
        """Perform TTS by taking a stream of characters and streaming output bytes."""
        service = self._get_service()

        # create an output wave file
        wav_file_name, wav_file = _mk_wave_file(
            self.output_directory, self.sample_rate_hertz
        )

        # split the input text and perform tts
        for chunk in _process_chunks(input):
            _LOGGER.debug("Riva TTS chunk: %s", chunk)

            # start riva tts streaming
            responses = service.synthesize_online(
                text=chunk,
                voice_name=self.voice_name,
                language_code=self.language_code,
                encoding=self.encoding.riva_pb2,
                sample_rate_hz=self.sample_rate_hertz,
            )

            # stream audio bytes out
            for resp in responses:
                audio = cast(bytes, resp.audio)
                if wav_file:
                    wav_file.writeframesraw(audio)
                yield audio

        # close the wave file when we are done
        if wav_file:
            wav_file.close()
            _LOGGER.debug("Riva TTS wrote file: %s", wav_file_name)

    async def atransform(
        self,
        input: AsyncIterator[TTSInputType],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncGenerator[TTSOutputType, None]:
        """Intercept async transforms and route them to the synchronous transform."""
        loop = asyncio.get_running_loop()
        input_queue: queue.Queue = queue.Queue()
        out_queue: asyncio.Queue = asyncio.Queue()

        async def _producer() -> None:
            """Produce input into the input queue."""
            async for val in input:
                input_queue.put_nowait(val)
            input_queue.put_nowait(_TRANSFORM_END)

        def _input_iterator() -> Iterator[TTSInputType]:
            """Iterate over the input_queue."""
            while True:
                try:
                    val = input_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if val == _TRANSFORM_END:
                    break
                yield val

        def _consumer() -> None:
            """Consume the input with transform."""
            for val in self.transform(_input_iterator()):
                out_queue.put_nowait(val)
            out_queue.put_nowait(_TRANSFORM_END)

        async def _consumer_coro() -> None:
            """Coroutine that wraps the consumer."""
            await loop.run_in_executor(None, _consumer)

        producer = loop.create_task(_producer())
        consumer = loop.create_task(_consumer_coro())

        while True:
            try:
                val = await asyncio.wait_for(out_queue.get(), 0.5)
            except asyncio.exceptions.TimeoutError:
                continue
            out_queue.task_done()

            if val is _TRANSFORM_END:
                break
            yield val

        await producer
        await consumer
