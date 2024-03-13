import logging
import os
from typing import Optional
from uuid import uuid4

import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import SecretStr
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class HuggingFaceTextToSpeechModelInference(BaseTool):
    """
    HuggingFace Text-to-Speech Model Inference.

    Requirements:

        - Environment variable ``HUGGINGFACE_API_KEY`` must be set,
          or passed as a named parameter to the constructor.
    """

    name: str = "openai_text_to_speech"
    description: str = "A wrapper around OpenAI Text-to-Speech API. "

    model: str
    file_extension: str

    api_url: str
    huggingface_api_key: SecretStr

    _HUGGINGFACE_API_KEY_ENV_NAME = "HUGGINGFACE_API_KEY"
    _HUGGINGFACE_API_URL_ROOT = "https://api-inference.huggingface.co/models"

    def __init__(
        self,
        model: str,
        file_extension: str,
        *,
        huggingface_api_key: Optional[SecretStr] = None,
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
            model=model,
            file_extension=file_extension,
            api_url=f"{self._HUGGINGFACE_API_URL_ROOT}/{model}",
            huggingface_api_key=huggingface_api_key,
        )

    def _run(
        self,
        query: str,
        output_base_name: Optional[str] = None,
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

        if output_base_name is None:
            output_base_name = str(uuid4())

        output_file = f"{output_base_name}.{self.file_extension}"

        try:
            with open(output_file, mode="xb") as f:
                f.write(audio_bytes)

        except FileExistsError:
            raise ValueError("Output name must be unique")

        except Exception as e:
            logger.error(f"Error occurred while creating file: {e}")
            raise

        return output_file
