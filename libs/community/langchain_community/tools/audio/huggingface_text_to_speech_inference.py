import os
from datetime import datetime
from enum import Enum
from typing import Optional

import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import SecretStr
from langchain_core.tools import BaseTool


class HuggingFaceSupportedAudioFormat(Enum):
    WAV = WAVE = WAVEFORM = "wav"


class HuggingFaceTextToSpeechModelInference(BaseTool):
    """
    HuggingFace Text-to-Speech Model Inference.

    Requirements:

        - Environment variable ``HUGGINGFACE_API_KEY`` must be set,
          or passed to the constructor.
    """

    name: str = "openai_text_to_speech"
    description: str = "A wrapper around OpenAI Text-to-Speech API. "

    model_name: str
    api_url: str
    huggingface_api_key: SecretStr
    format: HuggingFaceSupportedAudioFormat
    output_dir: str

    _DEFAULT_OUTPUT_DIR = "tts_output"
    _DEFAULT_OUTPUT_NAME = "output"
    _HUGGINGFACE_API_KEY_ENV_NAME = "HUGGINGFACE_API_KEY"
    _HUGGINGFACE_API_URL_ROOT = "https://api-inference.huggingface.co/models"

    def __init__(
        self,
        model_name: str,
        format: HuggingFaceSupportedAudioFormat,
        huggingface_api_key: Optional[SecretStr] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        if not huggingface_api_key:
            huggingface_api_key = SecretStr(
                os.getenv(self._HUGGINGFACE_API_KEY_ENV_NAME, "")
            )

        if (
            not huggingface_api_key
            or not huggingface_api_key.get_secret_value()
            or huggingface_api_key.get_secret_value() == ""
        ):
            raise ValueError(
                f"'{self._HUGGINGFACE_API_KEY_ENV_NAME}' must be or set or passed"
            )

        super().__init__(
            model_name=model_name,
            format=format,
            api_url=f"{self._HUGGINGFACE_API_URL_ROOT}/{model_name}",
            huggingface_api_key=huggingface_api_key,
            output_dir=output_dir if output_dir else self._DEFAULT_OUTPUT_DIR,
        )

    def _run(
        self,
        query: str,
        output_name: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.huggingface_api_key.get_secret_value()}"
            },
            json={"inputs": query},
        )

        audio_bytes = response.content

        output_name = (
            f"{output_name or self._default_output_name()}.{self.format.value}"
        )
        output_path = os.path.join(self.output_dir, output_name)

        if os.path.exists(output_path):
            raise ValueError("Output name must be unique")

        with open(output_path, mode="bx") as f:
            f.write(audio_bytes)

        return output_path

    def _default_output_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self._DEFAULT_OUTPUT_NAME}_{timestamp}"
