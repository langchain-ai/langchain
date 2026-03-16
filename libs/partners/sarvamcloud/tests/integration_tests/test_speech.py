"""Integration tests for speech services — requires SARVAM_API_SUBSCRIPTION_KEY."""

import pytest

from langchain_sarvamcloud.speech import SarvamSTT, SarvamTTS


@pytest.mark.requires("SARVAM_API_SUBSCRIPTION_KEY")
class TestSarvamSTTIntegration:
    def test_transcribe_hindi_audio(self) -> None:
        """Requires a real Hindi audio file at tests/fixtures/hindi_sample.wav."""
        import os

        audio_path = os.path.join(
            os.path.dirname(__file__), "../fixtures/hindi_sample.wav"
        )
        if not os.path.exists(audio_path):
            pytest.skip("Hindi audio fixture not found.")

        stt = SarvamSTT(model="saaras:v3", mode="transcribe")
        with open(audio_path, "rb") as f:
            result = stt.transcribe(f, language_code="hi-IN")
        assert "transcript" in result
        assert len(result["transcript"]) > 0


@pytest.mark.requires("SARVAM_API_SUBSCRIPTION_KEY")
class TestSarvamTTSIntegration:
    def test_synthesize_hindi_text(self) -> None:
        tts = SarvamTTS(speaker="Shubh", pace=1.0)
        result = tts.synthesize("नमस्ते", target_language_code="hi-IN")
        assert "audios" in result
        assert len(result["audios"]) > 0

    def test_synthesize_to_bytes(self) -> None:
        tts = SarvamTTS(speaker="Priya")
        audio_bytes = tts.synthesize_to_bytes("Hello", target_language_code="en-IN")
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0
