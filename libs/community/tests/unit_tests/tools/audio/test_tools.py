"""Test Audio Tools."""

import os
import tempfile
from unittest.mock import Mock, mock_open, patch

import pytest
from langchain_core.pydantic_v1 import SecretStr

from langchain_community.tools.audio import (
    HuggingFaceSupportedAudioFormat,
    HuggingFaceTextToSpeechModelInference,
)


def test_huggingface_tts_constructor() -> None:
    with pytest.raises(ValueError):
        os.environ.pop("HUGGINGFACE_API_KEY", None)
        HuggingFaceTextToSpeechModelInference(
            model="test/model", format=HuggingFaceSupportedAudioFormat.WAV
        )

    with pytest.raises(ValueError):
        HuggingFaceTextToSpeechModelInference(
            model="test/model",
            format=HuggingFaceSupportedAudioFormat.WAV,
            huggingface_api_key=SecretStr(""),
        )

    HuggingFaceTextToSpeechModelInference(
        model="test/model",
        format=HuggingFaceSupportedAudioFormat.WAV,
        huggingface_api_key=SecretStr("foo"),
    )

    os.environ["HUGGINGFACE_API_KEY"] = "foo"
    HuggingFaceTextToSpeechModelInference(
        model="test/model",
        format=HuggingFaceSupportedAudioFormat.WAV,
    )


def test_huggingface_tts_run_with_requests_mock() -> None:
    os.environ["HUGGINGFACE_API_KEY"] = "foo"

    format = HuggingFaceSupportedAudioFormat.WAV

    output_name = "test_output"
    input_query = "Dummy input"

    test_audio_content = b"test_audio_bytes"

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch("requests.post") as mock_inference,
        patch("builtins.open", mock_open()) as mock_file,
    ):
        expected_output_path = os.path.join(tmp_dir, f"{output_name}.wav")

        tts = HuggingFaceTextToSpeechModelInference(
            model="test/model",
            format=format,
            output_dir=tmp_dir,
        )

        # Mock the requests.post response
        mock_response = Mock()
        mock_response.content = test_audio_content
        mock_inference.return_value = mock_response

        output_path = tts._run(
            query=input_query,
            output_name=output_name,
        )

        assert output_path == expected_output_path

        mock_inference.assert_called_once_with(
            tts.api_url,
            headers={
                "Authorization": f"Bearer {tts.huggingface_api_key.get_secret_value()}"
            },
            json={"inputs": input_query},
        )

        mock_file.assert_called_once_with(output_path, mode="wb")
        mock_file.return_value.write.assert_called_once_with(test_audio_content)
