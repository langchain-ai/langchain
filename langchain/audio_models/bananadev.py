from typing import Dict, Optional

from pydantic import Extra, root_validator

from langchain.audio_models.base import AudioBase
from langchain.utils import get_from_dict_or_env


class AudioBanana(AudioBase):
    """Wrapper around Whisper models on Banana.

    To use, you should have the ``banana-dev`` python package installed,
    and the environment variable ``BANANA_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the call can be passed
    in, even if not explicitly saved on this class.
    """

    model_key: str = ""
    """model endpoint to use"""

    banana_api_key: Optional[str] = None

    max_chars: Optional[int] = None

    class Config:
        """Configuration for this pydantic config."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        banana_api_key = get_from_dict_or_env(
            values, "banana_api_key", "BANANA_API_KEY"
        )
        values["banana_api_key"] = banana_api_key
        return values

    def transcript(self, audio_path: str, task: str = "transcribe") -> str:
        """Call to Banana endpoint."""
        try:
            import banana_dev as banana
        except ImportError:
            raise ValueError(
                "Could not import banana-dev python package. "
                "Please install it with `pip install banana-dev`."
            )

        if task == "transcribe":
            mp3 = self._read_mp3_audio(audio_path)
            model_inputs = {"mp3BytesString": mp3}
            response = banana.run(self.banana_api_key, self.model_key, model_inputs)
            try:
                text = response["modelOutputs"][0]["text"]
                if not isinstance(text, str):
                    raise ValueError(f"Expected text to be a string, got {type(text)}")

            except KeyError:
                raise ValueError(
                    f"Response should be {'modelOutputs': [{'output': 'text'}]}."
                    f"Response was: {response}"
                )
        else:
            raise ValueError(
                f"Only task available is 'transcribe' for this model."
                f"Task was: {task}"
            )

        return text[: self.max_chars] if self.max_chars else text
