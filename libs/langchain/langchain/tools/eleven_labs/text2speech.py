import tempfile
from typing import Dict, Union

from langchain.pydantic_v1 import root_validator
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

    def _text2speech(self, text: str) -> str:
        speech = elevenlabs.generate(text=text, model=self.model)
        with tempfile.NamedTemporaryFile(mode="bx", suffix=".wav", delete=False) as f:
            f.write(speech)
        return f.name

    def _run(self, query: str) -> str:
        """Use the tool."""
        try:
            speech_file = self._text2speech(query)
            return speech_file
        except Exception as e:
            raise RuntimeError(f"Error while running ElevenLabsText2SpeechTool: {e}")

    def play(self, speech_file: str) -> None:
        """Play the text as speech."""
        with open(speech_file, mode="rb") as f:
            speech = f.read()

        elevenlabs.play(speech)

    def stream(self, query: str) -> None:
        """Stream the text as speech as it is generated.
        Play the text in your speakers."""
        speech_stream = elevenlabs.generate(text=query, model=self.model, stream=True)
        elevenlabs.stream(speech_stream)
