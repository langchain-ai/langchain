from typing import Dict, Optional

from langchain.audio_models.base import AudioBase
from langchain.utils import get_from_dict_or_env
from pydantic import Extra, root_validator


class WhisperAPI(AudioBase):
    """Wrapper around Whisper API.
    To use, you should have the ``openai`` python package installed,
    and the environment variable ``OPENAI_API_KEY`` set with your API key.
    Any parameters that are valid to be passed to the call can be passed
    in, even if not explicitly saved on this class.
    """

    audio_path: str

    model_key: str = ""
    """model endpoint to use"""

    openai_api_key: Optional[str] = None

    max_chars: Optional[int] = None

    model: str = "whisper-1"

    prompt: str = None

    response_format: str = "json"

    temperature: int = 0

    class Config:
        """Configuration for this pydantic config."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        values["openai_api_key"] = openai_api_key
        return values

    def transcribe(self) -> str:
        """Call to Whisper endpoint."""
        try:
            import openai
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        openai.api_key = model_key
        audio_file = open(audio_path, "rb")
        response = openai.Audio.transcribe(
            file=audio_file,
            model=model,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )
        try:
            text = response["text"]
            if not isinstance(text, str):
                raise ValueError(f"Expected text to be a string, got {type(text)}")

        except KeyError:
            raise ValueError(
                f"Response should be {'modelOutputs': [{'output': 'text'}]}."
                f"Response was: {response}"
            )

    def translate(self) -> str:
        """Call to Whisper endpoint."""
        try:
            import openai
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        openai.api_key = model_key
        audio_file = open(audio_path, "rb")
        response = openai.Audio.translate(
            file=audio_file,
            model=model,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )
        try:
            text = response["text"]
            if not isinstance(text, str):
                raise ValueError(f"Expected text to be a string, got {type(text)}")

        except KeyError:
            raise ValueError(
                f"Response should be {'modelOutputs': [{'output': 'text'}]}."
                f"Response was: {response}"
            )

        return text[: self.max_chars] if self.max_chars else text

    if __name__ == "__main__":
        test = WhisperAPI()
