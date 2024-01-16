"""An NVIDIA Riva ASR module that converts audio bytes to text."""
import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generator, Optional, cast

from langchain.pydantic_v1 import Field
from langchain_core.runnables import RunnableConfig

from .common import RivaAuthMixin, RivaBase, RivaCommonConfigMixin, _import_riva_client
from .stream import AudioStream

if TYPE_CHECKING:
    import riva.client

_LOGGER = logging.getLogger(__name__)
ASRInputType = AudioStream
ASROutputType = str


class RivaASR(
    RivaAuthMixin, RivaCommonConfigMixin, RivaBase[ASRInputType, ASROutputType]
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

    @property
    def config(self) -> "riva.client.StreamingRecognitionConfig":
        """Create and return the riva config object."""
        riva_client = _import_riva_client()
        return riva_client.StreamingRecognitionConfig(
            interim_results=False,
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

    def _transcribe(
        self,
        stream: AudioStream,
    ) -> Optional[str]:
        """Transcribe the audio bytes into a string with Riva stop when the person stops talking."""
        # create an output text generator with Riva
        if not stream.output:
            service = self._get_service()
            stream.output = service.streaming_response_generator(
                audio_chunks=stream,
                streaming_config=self.config,
            )

        # return the first valid result
        if stream.output:
            for response in stream.output:
                if not response.results:
                    continue
                for result in response.results:
                    if not result.alternatives:
                        continue
                    if result.is_final:
                        transcript = cast(str, result.alternatives[0].transcript)
                        _LOGGER.debug("Riva ASR returned: %s", transcript)
                        return transcript

        # no responses left in the output generator
        # if the input generator is complete, then so is the output
        if stream.hungup and stream.empty:
            stream.transcript_complete.set()

        return None

    def invoke(
        self,
        input: ASRInputType,
        _: Optional[RunnableConfig] = None,
        **__: Optional[Any],
    ) -> ASROutputType:
        """Perform ASR by ingesting an entire audio stream and returning the full transcript."""
        return "  ".join(iter(lambda: self._transcribe(input), None))

    def stream(
        self,
        input: ASRInputType,
        _: Optional[RunnableConfig] = None,
        **__: Optional[Any],
    ) -> Generator[ASROutputType, None, None]:
        """Perform ASR by transcoding the input audio stream and return the next identified prompt."""
        yield self._transcribe(input)

    async def astream(
        self,
        input: ASRInputType,
        _: Optional[RunnableConfig] = None,
    ) -> AsyncGenerator[ASROutputType, None]:
        """Intercept async streams and route them to the synchronous transform in a thread."""
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(None, self._transcribe, input)
        if out:
            yield out
