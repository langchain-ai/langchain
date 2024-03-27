from datetime import datetime
from enum import Enum
from os import path
from typing import Any, Optional

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

    model: OpenAISpeechModel
    voice: OpenAISpeechVoice
    speed: float
    format: OpenAISpeechFormat
    output_dir: str

    _DEFAULT_SPEED = 1.0
    _DEFAULT_OUTPUT_DIR = "tts_output"
    _DEFAULT_OUTPUT_NAME_PREFIX = "output"
    # https://platform.openai.com/docs/api-reference/audio/createSpeech#audio-createspeech-input
    _MAX_CHARACTER_LENGTH = 4096
    _MIN_SPEED = 0.25
    _MAX_SPEED = 4.0

    @validator("speed", always=True)
    def check_speed_constraints(cls, speed: float) -> float:
        if not (cls._MIN_SPEED <= speed <= cls._MAX_SPEED):
            raise ValueError(
                "Value must be greater or equal than 0.25 and less than or equal to 4.0"
            )
        return speed

    def __init__(
        self,
        model: Optional[OpenAISpeechModel] = None,
        voice: Optional[OpenAISpeechVoice] = None,
        speed: Optional[float] = None,
        format: Optional[OpenAISpeechFormat] = None,
        output_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes private fields."""
        super().__init__(
            model=model if model is not None else OpenAISpeechModel.DEFAULT,
            voice=voice if voice is not None else OpenAISpeechVoice.DEFAULT,
            speed=speed if speed is not None else OpenAITextToSpeechTool._DEFAULT_SPEED,
            format=format if format is not None else OpenAISpeechFormat.DEFAULT,
            output_dir=output_dir
            if output_dir is not None
            else OpenAITextToSpeechTool._DEFAULT_OUTPUT_DIR,
            **kwargs,
        )

    def _run(
        self,
        input: str,
        output_name: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if len(input) > self._MAX_CHARACTER_LENGTH:
            # TODO: break up input string and concat results into one file
            raise ValueError(
                f"Max input character length is {self._MAX_CHARACTER_LENGTH}."
            )

        output_name = (
            output_name if output_name is not None else self._default_output_name()
        )
        output_path = path.join(self.output_dir, f"{output_name}.{self.format.value}")

        if path.exists(output_path):
            raise ValueError(
                "File already exists at output path,"
                f"provide a unique path: '{output_path}'"
            )

        response = openai.audio.speech.create(
            model=self.model.value,
            voice=self.voice.value,
            speed=self.speed,
            response_format=self.format.value,
            input=input,
        )

        response.stream_to_file(output_path)

        return output_path

    def _default_output_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self._DEFAULT_OUTPUT_NAME_PREFIX}_{timestamp}"
