from typing import Optional

from pydantic import Extra

from langchain.audio_models.base import AudioBase


class Whisper(AudioBase):
    """Wrapper around Whisper API.
    To use, you should have the ``openai`` python package installed,
    and the environment variable ``OPENAI_API_KEY`` set with your API key.
    Any parameters that are valid to be passed to the call can be passed
    in, even if not explicitly saved on this class.
    """

    model_key: str = ""
    prompt: Optional[str] = ""
    language: Optional[str] = ""
    max_chars: Optional[int] = None
    model: Optional[str] = "whisper-1"
    temperature: Optional[float] = 0.0
    response_format: Optional[str] = "json"

    class Config:
        """Configuration for this pydantic config."""

        extra = Extra.forbid

    def transcript(self, audio_path: str, task: str = "transcribe") -> str:
        """Call to Whisper transcribe endpoint."""

        try:
            import openai
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        if task == "transcribe":
            openai.api_key = self.model_key
            audio_file = open(audio_path, "rb")
            response = openai.Audio.transcribe(
                file=audio_file,
                model=self.model,
                prompt=self.prompt,
                language=self.language,
                temperature=self.temperature,
                response_format=self.response_format,
            )

        if task == "translate":
            openai.api_key = self.model_key
            audio_file = open(audio_path, "rb")
            response = openai.Audio.translate(
                file=audio_file,
                model=self.model,
                prompt=self.prompt,
                temperature=self.temperature,
                response_format=self.response_format,
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
