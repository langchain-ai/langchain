from unittest.mock import mock_open, patch

import pytest

from langchain_groq.transcription import TranscriptionGroq
import httpx

def test_init_without_api_key_raises():
    # Clear env variable for test
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(
            ValueError, match="GROQ_API_KEY environment variable not set"
        ):
            TranscriptionGroq()


@patch("builtins.open", new_callable=mock_open, read_data=b"audio data")
@patch("httpx.post")
def test_transcribe_success(mock_post, mock_file):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"text": "Hello world"}

    transcriber = TranscriptionGroq(api_key="fakekey")
    result = transcriber.transcribe("fake_path.mp3")

    mock_file.assert_called_once_with("fake_path.mp3", "rb")
    assert result == "Hello world"


@patch("builtins.open", new_callable=mock_open, read_data=b"audio data")
@patch("httpx.post")
def test_transcribe_failure(mock_post, mock_file):
    mock_post.return_value.status_code = 400
    mock_post.return_value.text = "Bad request"

    transcriber = TranscriptionGroq(api_key="fakekey")
    with pytest.raises(RuntimeError, match="Transcription failed: 400 - Bad request"):
        transcriber.transcribe("fake_path.mp3")
