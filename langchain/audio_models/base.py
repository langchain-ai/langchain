"""Base class for text to audio models."""
import base64
import pathlib
from abc import ABC, abstractmethod
from io import BytesIO

from pydantic import BaseModel


class AudioBase(BaseModel, ABC):
    @staticmethod
    def _read_mp3_audio(audio_path: str) -> str:
        """Read audio file."""
        if not pathlib.Path(audio_path).exists():
            raise ValueError(f"Can't find audio file at {audio_path}")
        if not audio_path.endswith(".mp3"):
            raise ValueError("Only mp3 files are supported.")
        with open(audio_path, "rb") as file:
            mp3bytes = BytesIO(file.read())
        return base64.b64encode(mp3bytes.getvalue()).decode("ISO-8859-1")

    @abstractmethod
    def transcript(self, audio_path: str, task: str) -> str:
        """Transcribe audio file."""
