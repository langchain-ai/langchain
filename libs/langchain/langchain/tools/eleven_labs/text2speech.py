from typing import Dict, Union

from langchain.pydantic_v1 import root_validator
from langchain.tools.audio_utils import save_audio, load_audio
from langchain.tools.base import BaseTool
from langchain.tools.eleven_labs.models import ElevenLabsModel
from langchain.utils import get_from_dict_or_env

try:
    import elevenlabs

except ImportError:
    raise ImportError(
        "elevenlabs is not installed. " "Run `pip install elevenlabs` to install."
    )


class ElevenLabsText2SpeechTool(BaseTool):
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

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        _ = get_from_dict_or_env(values, "eleven_api_key", "ELEVEN_API_KEY")

        return values

    def _text2speech(self, text: str) -> bytes:
        speech = elevenlabs.generate(text=text, model=self.model)
        return speech

    def _run(self, query: str) -> str:
        """Use the tool."""
        try:
            speech = self._text2speech(query)
            self.play(speech)
            return "Speech has been generated"
        except Exception as e:
            raise RuntimeError(f"Error while running ElevenLabsText2SpeechTool: {e}")

    def play(self, speech) -> None:
        """Play the text as speech."""
        elevenlabs.play(speech)

    def stream(self, query: str) -> None:
        """Stream the text as speech as it is generated.
        Play the text in your speakers."""
        speech_stream = elevenlabs.generate(text=query, model=self.model, stream=True)
        elevenlabs.stream(speech_stream)
        
    def generate_and_save(self, query: str) -> str:
        """Save the text as speech to a temporary file."""
        speech = self._text2speech(query)
        path = save_audio(speech)
        return path

    def load_and_play(self, path: str) -> None:
        """Load the text as speech from a temporary file."""
        speech = load_audio(path)
        self.play(speech)