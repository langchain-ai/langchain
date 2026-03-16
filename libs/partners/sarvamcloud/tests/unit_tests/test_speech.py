"""Unit tests for Sarvam AI speech services."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest


class TestSarvamSTT:
    @pytest.fixture()
    def stt(self) -> "SarvamSTT":
        from langchain_sarvamcloud.speech import SarvamSTT

        mock_client = MagicMock()
        mock_client.speech_to_text.transcribe.return_value = {
            "request_id": "req-123",
            "transcript": "namaste",
            "language_code": "hi-IN",
            "language_probability": 0.99,
        }
        with patch("sarvamai.SarvamAI", return_value=mock_client):
            stt = SarvamSTT(
                model="saaras:v3",
                mode="transcribe",
                api_subscription_key="test-key",  # type: ignore[arg-type]
            )
        stt._client = mock_client
        return stt

    def test_transcribe_returns_transcript(self, stt: "SarvamSTT") -> None:
        audio = BytesIO(b"fake-audio-data")
        result = stt.transcribe(audio, language_code="hi-IN")
        assert "transcript" in result
        assert result["transcript"] == "namaste"

    def test_transcribe_passes_correct_params(self, stt: "SarvamSTT") -> None:
        audio = BytesIO(b"fake-audio-data")
        stt.transcribe(audio, language_code="ta-IN", mode="translate")
        call_kwargs = stt._client.speech_to_text.transcribe.call_args[1]
        assert call_kwargs["language_code"] == "ta-IN"
        assert call_kwargs["mode"] == "translate"
        assert call_kwargs["model"] == "saaras:v3"

    def test_default_model_is_saaras_v3(self) -> None:
        from langchain_sarvamcloud.speech import SarvamSTT

        with patch("sarvamai.SarvamAI"):
            stt = SarvamSTT(api_subscription_key="key")  # type: ignore[arg-type]
        assert stt.model == "saaras:v3"


class TestSarvamTTS:
    @pytest.fixture()
    def tts(self) -> "SarvamTTS":
        from langchain_sarvamcloud.speech import SarvamTTS

        mock_client = MagicMock()
        import base64

        mock_client.text_to_speech.convert.return_value = {
            "request_id": "req-456",
            "audios": [base64.b64encode(b"fake-wav-data").decode()],
        }
        with patch("sarvamai.SarvamAI", return_value=mock_client):
            tts = SarvamTTS(
                speaker="shubh",
                pace=1.0,
                api_subscription_key="test-key",  # type: ignore[arg-type]
            )
        tts._client = mock_client
        return tts

    def test_synthesize_returns_audio(self, tts: "SarvamTTS") -> None:
        result = tts.synthesize("Hello", target_language_code="en-IN")
        assert "audios" in result
        assert len(result["audios"]) > 0

    def test_synthesize_to_bytes_returns_bytes(self, tts: "SarvamTTS") -> None:
        audio_bytes = tts.synthesize_to_bytes("Hello", target_language_code="en-IN")
        assert isinstance(audio_bytes, bytes)
        assert audio_bytes == b"fake-wav-data"

    def test_synthesize_passes_speaker(self, tts: "SarvamTTS") -> None:
        tts.synthesize("Hi", target_language_code="hi-IN", speaker="priya")
        call_kwargs = tts._client.text_to_speech.convert.call_args[1]
        assert call_kwargs["speaker"] == "priya"

    def test_synthesize_passes_pace(self, tts: "SarvamTTS") -> None:
        tts.synthesize("Hi", target_language_code="hi-IN", pace=1.5)
        call_kwargs = tts._client.text_to_speech.convert.call_args[1]
        assert call_kwargs["pace"] == 1.5

    def test_default_model_is_bulbul_v3(self) -> None:
        from langchain_sarvamcloud.speech import SarvamTTS

        with patch("sarvamai.SarvamAI"):
            tts = SarvamTTS(api_subscription_key="key")  # type: ignore[arg-type]
        assert tts.model == "bulbul:v3"

    def test_synthesize_to_bytes_raises_on_empty_audio(self, tts: "SarvamTTS") -> None:
        tts._client.text_to_speech.convert.return_value = {
            "request_id": "req-789",
            "audios": [],
        }
        with pytest.raises(ValueError, match="No audio returned"):
            tts.synthesize_to_bytes("test", target_language_code="en-IN")


class TestSarvamBatchSTT:
    @pytest.fixture()
    def batch_stt(self) -> "SarvamBatchSTT":
        from langchain_sarvamcloud.speech import SarvamBatchSTT

        mock_client = MagicMock()
        mock_client.speech_to_text_job.create_job.return_value = {
            "job_id": "batch-job-123",
            "job_state": "Accepted",
        }
        mock_client.speech_to_text_job.get_status.return_value = {
            "job_id": "batch-job-123",
            "job_state": "Completed",
            "total_files": 2,
            "successful_files_count": 2,
            "failed_files_count": 0,
            "job_details": [
                {"file": "audio1.mp3", "output": "Hello world", "error": None}
            ],
        }
        with patch("sarvamai.SarvamAI", return_value=mock_client):
            batch = SarvamBatchSTT(
                api_subscription_key="test-key"  # type: ignore[arg-type]
            )
        batch._client = mock_client
        return batch

    def test_create_job_returns_job_id(self, batch_stt: "SarvamBatchSTT") -> None:
        job_id = batch_stt.create_job(model="saaras:v3")
        assert job_id == "batch-job-123"

    def test_get_status_returns_completed(self, batch_stt: "SarvamBatchSTT") -> None:
        status = batch_stt.get_status("batch-job-123")
        assert status["job_state"] == "Completed"
        assert status["successful_files_count"] == 2

    def test_wait_for_completion_returns_when_done(
        self, batch_stt: "SarvamBatchSTT"
    ) -> None:
        batch_stt._client.speech_to_text_job.get_status.return_value = {
            "job_state": "Completed",
            "job_id": "batch-job-123",
        }
        result = batch_stt.wait_for_completion("batch-job-123", poll_interval=0.0)
        assert result["job_state"] == "Completed"
