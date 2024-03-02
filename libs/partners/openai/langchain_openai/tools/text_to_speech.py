from enum import Enum
from os import path
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

import openai
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import validator
from langchain_core.tools import BaseTool


class OpenAISpeechModel(Enum):
    """
    Models available for OpenAI TTS.
    Docs: https://platform.openai.com/docs/models/tts
    """

    TTS_1 = DEFAULT = "tts-1"
    TTS_1_HD = "tts-1-hd"


class OpenAISpeechVoice(Enum):
    """
    Voices available for OpenAI TTS.
    #audio-createspeech-voice
    Docs: https://platform.openai.com/docs/api-reference/audio/createSpeech
    """

    ALLOY = DEFAULT = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"


class OpenAISpeechFormat(Enum):
    """
    Response formats available for OpenAI TTS.
    #audio-createspeech-response_format
    Docs: https://platform.openai.com/docs/api-reference/audio/createSpeech
    """

    MP3 = DEFAULT = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"


class OpenAITextToSpeechTool(BaseTool):
    """
    Tool that queries the OpenAI Text to Speech API.

    Supports many languages (https://platform.openai.com/docs/guides/text-to-speech/supported-languages),
    just provide input in language of choice.
    """

    name: str = "openai_text_to_speech"
    description: str = "A wrapper around OpenAI Text-to-Speech API. "

    _model: OpenAISpeechModel
    _voice: OpenAISpeechVoice
    _speed: float
    _format: OpenAISpeechFormat

    _DEFAULT_SPEED = 1.0
    _DEFAULT_OUTPUT_DIR = "./tts/"
    _DEFAULT_OUTPUT_NAME_PREFIX = "output"
    # https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-input
    _MAX_CHARACTER_LENGTH = 4096
    _MIN_SPEED = 0.25
    _MAX_SPEED = 4.0

    @validator("_speed")
    def validate_my_float(cls, v: float) -> float:
        if not (
            OpenAITextToSpeechTool._MIN_SPEED <= v <= OpenAITextToSpeechTool._MAX_SPEED
        ):
            raise ValueError(
                "Value must be greater or equal than 0.25 and less than or equal to 4.0"
            )
        return v

    def __init__(
        self,
        model: Optional[OpenAISpeechModel] = None,
        voice: Optional[OpenAISpeechVoice] = None,
        speed: Optional[float] = None,
        format: Optional[OpenAISpeechFormat] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes private fields."""
        super().__init__(**kwargs)

        self._model = model or OpenAISpeechModel.DEFAULT
        self._voice = voice or OpenAISpeechVoice.DEFAULT
        self._speed = speed or OpenAITextToSpeechTool._DEFAULT_SPEED
        self._format = format or OpenAISpeechFormat.DEFAULT

    def _run(
        self,
        input: str,
        output_dir: Optional[str] = None,
        output_name: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Path:
        """Use the tool."""
        if len(input) > self._MAX_CHARACTER_LENGTH:
            # TODO: break up input string and concat results into one file
            raise ValueError(
                f"Max input character length is {self._MAX_CHARACTER_LENGTH}."
            )

        output_dir = output_dir or OpenAITextToSpeechTool._DEFAULT_OUTPUT_DIR
        output_name = output_name or self._default_output_name()

        output_path = Path(output_dir) / f"{output_name}.{self._format}"

        if path.exists(output_path):
            raise ValueError(
                "File already exists at output path,"
                f"provide a unique path: '{output_path}'"
            )

        response = openai.audio.speech.create(
            model=self._model.value,
            voice=self._voice.value,
            speed=self._speed,
            response_format=self._format.value,
            input=input,
        )

        response.stream_to_file(output_path)

        return output_path

    def _default_output_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (
            f"{OpenAITextToSpeechTool._DEFAULT_OUTPUT_NAME_PREFIX}_{timestamp}"
        )
