import os
from typing import Optional

import requests


class TranscriptionGroq:
    """
    Transcribe audio using Groq Whisper models.
    """

    def __init__(
        self, model: str = "whisper-large-v3-turbo", api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.endpoint = "https://api.groq.com/openai/v1/audio/transcriptions"

        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        with open(audio_path, "rb") as audio_file:
            files = {
                "file": (audio_path, audio_file, "audio/mpeg"),
                "model": (None, self.model),
            }
            if language:
                files["language"] = (None, language)

            response = requests.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
            )

        if response.status_code == 200:
            return response.json()["text"]
        else:
            raise RuntimeError(
                f"Transcription failed: {response.status_code} - {response.text}"
            )
