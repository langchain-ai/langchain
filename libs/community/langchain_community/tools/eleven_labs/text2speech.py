import tempfile
from enum import Enum
from typing import Any, Dict, Optional, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator


def _import_elevenlabs() -> Any:
    try:
        import elevenlabs
    except ImportError as e:
        raise ImportError(
            "Cannot import elevenlabs, please install `pip install elevenlabs`."
        ) from e
    return elevenlabs


class ElevenLabsModel(str, Enum):
    """Models available for Eleven Labs Text2Speech."""

    MULTI_LINGUAL = "eleven_multilingual_v2"
    MULTI_LINGUAL_FLASH = "eleven_flash_v2_5"
    MONO_LINGUAL = "eleven_flash_v2"


class ElevenLabsText2SpeechTool(BaseTool):  # type: ignore[override]
    """Tool that queries the Eleven Labs Text2Speech API.

    In order to set this up, follow instructions at:
    https://elevenlabs.io/docs
    """

    model: Union[ElevenLabsModel, str] = ElevenLabsModel.MULTI_LINGUAL
    voice: str = "JBFqnCBsd6RMkjVDRZzb"

    name: str = "eleven_labs_text2speech"
    description: str = (
        "A wrapper around Eleven Labs Text2Speech. "
        "Useful for when you need to convert text to speech. "
        "It supports more than 30 languages, including English, German, Polish, "
        "Spanish, Italian, French, Portuguese, and Hindi. "
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        _ = get_from_dict_or_env(values, "elevenlabs_api_key", "ELEVENLABS_API_KEY")

        return values

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        elevenlabs = _import_elevenlabs()
        client = elevenlabs.client.ElevenLabs()
        try:
            speech = client.text_to_speech.convert(
                text=query,
                model_id=self.model,
                voice_id=self.voice,
                output_format="mp3_44100_128",
            )
            with tempfile.NamedTemporaryFile(
                mode="bx", suffix=".mp3", delete=False
            ) as f:
                f.write(speech)
            return f.name
        except Exception as e:
            raise RuntimeError(f"Error while running ElevenLabsText2SpeechTool: {e}")

    def play(self, speech_file: str) -> None:
        """Play the text as speech."""
        elevenlabs = _import_elevenlabs()
        with open(speech_file, mode="rb") as f:
            speech = f.read()

        elevenlabs.play(speech)

    def stream_speech(self, query: str) -> None:
        """Stream the text as speech as it is generated.
        Play the text in your speakers."""
        elevenlabs = _import_elevenlabs()
        client = elevenlabs.client.ElevenLabs()
        speech_stream = client.text_to_speech.convert_as_stream(
            text=query, model_id=self.model, voice_id=self.voice
        )
        elevenlabs.stream(speech_stream)
