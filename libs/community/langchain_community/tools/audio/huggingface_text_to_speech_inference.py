import logging
import os
import uuid
from datetime import datetime
from typing import Callable, Literal, Optional

import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class HuggingFaceTextToSpeechModelInference(BaseTool):
    """HuggingFace Text-to-Speech Model Inference.

    Requirements:
        - Environment variable ``HUGGINGFACE_API_KEY`` must be set,
          or passed as a named parameter to the constructor.
    """

    name: str = "openai_text_to_speech"
    """Name of the tool."""
    description: str = "A wrapper around OpenAI Text-to-Speech API. "
    """Description of the tool."""

    model: str
    """Model name."""
    file_extension: str
    """File extension of the output audio file."""
    destination_dir: str
    """Directory to save the output audio file."""
    file_namer: Callable[[], str]
    """Function to generate unique file names."""

    api_url: str
    huggingface_api_key: SecretStr

    _HUGGINGFACE_API_KEY_ENV_NAME: str = "HUGGINGFACE_API_KEY"
    _HUGGINGFACE_API_URL_ROOT: str = "https://api-inference.huggingface.co/models"

    def __init__(
        self,
        model: str,
        file_extension: str,
        *,
        destination_dir: str = "./tts",
        file_naming_func: Literal["uuid", "timestamp"] = "uuid",
        huggingface_api_key: Optional[SecretStr] = None,
        _HUGGINGFACE_API_KEY_ENV_NAME: str = "HUGGINGFACE_API_KEY",
        _HUGGINGFACE_API_URL_ROOT: str = "https://api-inference.huggingface.co/models",
    ) -> None:
        if not huggingface_api_key:
            huggingface_api_key = SecretStr(
                os.getenv(_HUGGINGFACE_API_KEY_ENV_NAME, "")
            )

        if (
            not huggingface_api_key
            or not huggingface_api_key.get_secret_value()
            or huggingface_api_key.get_secret_value() == ""
        ):
            raise ValueError(
                f"'{_HUGGINGFACE_API_KEY_ENV_NAME}' must be or set or passed"
            )

        if file_naming_func == "uuid":
            file_namer = lambda: str(uuid.uuid4())  # noqa: E731
        elif file_naming_func == "timestamp":
            file_namer = lambda: str(int(datetime.now().timestamp()))  # noqa: E731
        else:
            raise ValueError(
                f"Invalid value for 'file_naming_func': {file_naming_func}"
            )

        super().__init__(  # type: ignore[call-arg]
            model=model,
            file_extension=file_extension,
            api_url=f"{_HUGGINGFACE_API_URL_ROOT}/{model}",
            destination_dir=destination_dir,
            file_namer=file_namer,
            huggingface_api_key=huggingface_api_key,
            _HUGGINGFACE_API_KEY_ENV_NAME=_HUGGINGFACE_API_KEY_ENV_NAME,
            _HUGGINGFACE_API_URL_ROOT=_HUGGINGFACE_API_URL_ROOT,
        )

    def _run(
        self,
        query: str,
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

        try:
            os.makedirs(self.destination_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory '{self.destination_dir}': {e}")
            raise

        output_file = os.path.join(
            self.destination_dir,
            f"{str(self.file_namer())}.{self.file_extension}",
        )

        try:
            with open(output_file, mode="xb") as f:
                f.write(audio_bytes)
        except FileExistsError:
            raise ValueError("Output name must be unique")
        except Exception as e:
            logger.error(f"Error occurred while creating file: {e}")
            raise

        return output_file
