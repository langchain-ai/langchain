import logging
import os
import uuid
from datetime import datetime
from typing import Callable, Literal, Optional

import openai
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class OpenAITextToSpeechTool(BaseTool):
    """
    Tool that queries the OpenAI Text to Speech API.

    Supports many languages (https://platform.openai.com/docs/guides/text-to-speech/supported-languages),
    just provide input in language of choice.

    Models available for OpenAI TTS.
    Docs: https://platform.openai.com/docs/models/tts

    Voices available for OpenAI TTS.
    Docs: https://platform.openai.com/docs/api-reference/audio/createSpeech
    """

    name: str = "openai_text_to_speech"
    """Name of the tool."""
    description: str = "A wrapper around OpenAI Text-to-Speech API. "
    """Description of the tool."""

    model: str
    """Model name."""
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    """Voice name."""
    speed: float
    """Audio speed."""
    file_extension: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
    """File extension of the output audio file."""
    destination_dir: str
    """Directory to save the output audio file."""
    file_namer: Callable[[], str]
    """Function to generate unique file names."""

    # https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-input
    _MAX_CHARACTER_LENGTH = 4096

    def __init__(
        self,
        model: str,
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        *,
        file_extension: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3",
        speed: float = 1.0,
        destination_dir: str = "./tts",
        file_naming_func: Literal["uuid", "timestamp"] = "uuid",
    ) -> None:
        if file_naming_func == "uuid":
            file_namer = lambda: str(uuid.uuid4())  # noqa: E731
        elif file_naming_func == "timestamp":
            file_namer = lambda: str(int(datetime.now().timestamp()))  # noqa: E731
        else:
            raise ValueError(
                f"Invalid value for 'file_naming_func': {file_naming_func}"
            )

        super().__init__(
            model=model,
            voice=voice,
            speed=speed,
            file_extension=file_extension,
            destination_dir=destination_dir,
            file_namer=file_namer,
        )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if len(query) > self._MAX_CHARACTER_LENGTH:
            # TODO: break up input string and concat results into one file
            raise ValueError(
                f"Max input character length is {self._MAX_CHARACTER_LENGTH}."
            )

        response = openai.audio.speech.create(
            model=self.model,
            voice=self.voice,
            speed=self.speed,
            response_format=self.file_extension,
            input=query,
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
