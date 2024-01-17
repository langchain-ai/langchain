"""An NVIDIA Riva TTS module that converts text into audio bytes."""
import logging
import pathlib
import tempfile
import wave
from typing import TYPE_CHECKING, Any, Generator, Iterator, Optional, Tuple, Union, cast

from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import Field, validator
from langchain_core.runnables import RunnableConfig

from .common import RivaAuthMixin, RivaBase, RivaCommonConfigMixin, _import_riva_client

if TYPE_CHECKING:
    import riva.client

_LOGGER = logging.getLogger(__name__)
_MAX_TEXT_LENGTH = 400
_SENTENCE_TERMINATORS = ("\n", ".", "!", "?", "¡", "¿")

# pylint: disable-next=invalid-name
TTSInputType = Union[str, AnyMessage, PromptValue]
TTSOutputType = bytes


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


def _coerce_string(val: TTSInputType) -> str:
    """Attempt to coerce the input value to a string.

    This is particularly useful for converting LangChain message to strings.
    """
    if isinstance(val, PromptValue):
        return val.to_string()
    if isinstance(val, BaseMessage):
        return str(val.content)
    return str(val)


def _process_chunks(inputs: Iterator[TTSInputType]) -> Generator[str, None, None]:
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

    # return any remaining buffer
    if buffer:
        yield buffer


class RivaTTS(
    RivaAuthMixin, RivaCommonConfigMixin, RivaBase[TTSInputType, TTSOutputType]
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
            "A null value indicates that wave files should not be saved. This is useful for debugging purposes."
        ),
    )

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
        self, input: TTSInputType, _: RunnableConfig | None = None
    ) -> TTSOutputType:
        """Perform TTS by taking a string and outputting the entire audio file."""
        return b"".join(self.transform(iter([input])))

    def transform(
        self,
        input: Iterator[TTSInputType],
        _: Optional[RunnableConfig] = None,
        **__: Optional[Any],
    ) -> Generator[TTSOutputType, None, None]:
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
