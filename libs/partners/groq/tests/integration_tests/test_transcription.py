import os

import pytest

from langchain_groq import TranscriptionGroq


@pytest.mark.skipif(
    not os.path.exists("libs/partners/groq/tests/assets/hello.mp3"),
    reason="Missing test audio",
)
@pytest.mark.skipif("GROQ_API_KEY" not in os.environ, reason="GROQ_API_KEY not set")
def test_transcription_real():
    transcriber = TranscriptionGroq(model="whisper-large-v3-turbo")
    audio_path = "libs/partners/groq/tests/assets/hello.mp3"
    result = transcriber.transcribe(audio_path)
    assert isinstance(result, str)
    assert "hello" in result.lower()
