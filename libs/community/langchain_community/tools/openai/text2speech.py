import tempfile
from enum import Enum
from typing import Any, Dict, Optional, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator

import torch
from dataclasses import dataclass, field
from openai import OpenAI

class OpenAIText2SpeechTool(BaseTool):  # type: ignore[override]
    """Tool that queries OpenAI Speech API.
    """

    name: str = "openai_text2speech"
    description: str = (
        "A wrapper around OpenAI Text2Speech. "
        "Useful for when you need to convert text to speech. "
        "It supports multiple languages, including English, German, Polish, "
        "Spanish, Italian, French, Portuguese, and Hindi. "
    )

    base_url: str
    api_key: str
    model_id: str
    voice: str    

    def __init__(self, model_id: str, voice: str, base_url: str, api_key: str) -> None:
        model_id = model_id
        voice = voice
        base_url = base_url
        api_key = api_key

        super().__init__(  # type: ignore[call-arg]
            model_id=model_id,
            voice=voice,
            base_url=base_url,
            api_key=api_key
        )

    def _run(
        self, 
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""


        try:
            out_file_name = "tts-output.mp3"
            client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key
            )
            with client.audio.speech.with_streaming_response.create(
                model=self.model_id, 
                voice=self.voice,
                input=query,
                response_format="mp3"
            ) as response:
                response.stream_to_file(out_file_name)

            return out_file_name
        except Exception as e:
            raise RuntimeError(f"Error while running OpenAIText2SpeechTool: {e}")

    def play(self, speech_file: str) -> None:
        """Play the text as speech."""

        import sounddevice as sd
        import soundfile as sf
        import numpy as np

        _, r = sf.read(speech_file, dtype='float32')

        try:
            from transformers.pipelines.audio_utils import ffmpeg_read
        except ImportError as exc:
            raise ImportError(
                "Could not import ffmpeg-python python package. "
                "Please install it with `pip install ffmpeg-python`."
            ) from exc

        audio_decoded = None
        with open(speech_file, "rb") as f:
            content = f.read()
            audio_decoded = ffmpeg_read(content, r)
        d = np.frombuffer(audio_decoded, dtype=np.float32)

        sd.play(d,r)
        sd.wait()

    def stream_speech(self, query: str) -> None:
        """Stream the text as speech as it is generated.
        Play the text in your speakers."""

        import numpy as np

        try:
            import pyaudio
        except ImportError as exc:
            raise ImportError(
                    "Could not import pyaudio python package. "
                    "Please install it with `pip install pyaudio`."
            ) from exc

        try:
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
            player = pyaudio.PyAudio().open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True
            )

            with client.audio.speech.with_streaming_response.create(
                model=self.model_id,
                voice=self.voice,
                input=query,
                response_format="pcm"
            ) as response:

                for chunk in response.iter_bytes(chunk_size=1024):
                    d = np.frombuffer(chunk, dtype=np.int16)
                    player.write(chunk)
        except Exception as e:
            raise RuntimeError(f"Error while running OpenAIText2SpeechTool: {e}")
