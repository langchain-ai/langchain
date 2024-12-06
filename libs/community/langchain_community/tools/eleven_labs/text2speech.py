import tempfile
from enum import Enum
from typing import Any, Dict, Optional, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator


def _import_elevenlabs() -> Any:
    try:
        from elevenlabs import play, stream
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise ImportError(
            "Cannot import elevenlabs, please install `pip install elevenlabs`."
        ) from e
    return ElevenLabs, play, stream


class ElevenLabsModel(str, Enum):
    """Models available for Eleven Labs Text2Speech."""

    MULTI_LINGUAL = "eleven_multilingual_v1"
    MONO_LINGUAL = "eleven_monolingual_v1"


class ElevenLabsText2SpeechTool(BaseTool):  # type: ignore[override]
    """Tool that queries the Eleven Labs Text2Speech API.

    In order to set this up, follow instructions at:
    https://docs.elevenlabs.io/welcome/introduction
    """

    model: Union[ElevenLabsModel, str] = ElevenLabsModel.MULTI_LINGUAL

    name: str = "eleven_labs_text2speech"
    description: str = (
        "A wrapper around Eleven Labs Text2Speech. "
        "Useful for when you need to convert text to speech. "
        "It supports multiple languages, including English, German, Polish, "
        "Spanish, Italian, French, Portuguese, and Hindi. "
    )
    eleven_api_key: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that API key exists in environment or is provided directly."""
        if not values.get("eleven_api_key"):
            values["eleven_api_key"] = get_from_dict_or_env(
                {}, "eleven_api_key", "ELEVEN_API_KEY"
            )
        return values

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        ElevenLabs, _, _ = _import_elevenlabs()
        try:
            # Use the API key provided during initialization
            api_key = self.eleven_api_key or get_from_dict_or_env(
                {}, "eleven_api_key", "ELEVEN_API_KEY"
            )
            client = ElevenLabs(api_key=api_key)
            speech = client.generate(text=query, model=self.model)

            # Write the generator audio output to a file
            with tempfile.NamedTemporaryFile(
                mode="bx", suffix=".wav", delete=False
            ) as f:
                for chunk in speech:
                    f.write(chunk)
            return f.name
        except Exception as e:
            raise RuntimeError(f"Error while running ElevenLabsText2SpeechTool: {e}")

    def play(self, speech_file: str) -> None:
        """Play the text as speech."""
        _, play, _ = _import_elevenlabs()
        with open(speech_file, mode="rb") as f:
            speech = f.read()

        play(speech)

    def stream_speech(self, query: str) -> None:
        """Stream the text as speech as it is generated.
        Play the text in your speakers."""
        ElevenLabs, _, stream = _import_elevenlabs()
        try:
            # Use the API key provided during initialization
            api_key = self.eleven_api_key or get_from_dict_or_env(
                {}, "eleven_api_key", "ELEVEN_API_KEY"
            )
            client = ElevenLabs(api_key=api_key)
            # Generate the audio stream
            speech_stream = client.generate(text=query, model=self.model, stream=True)
            # Stream the generated audio
            stream(speech_stream)
        except Exception as e:
            raise RuntimeError(f"Error while streaming ElevenLabsText2SpeechTool: {e}")
