"""Test Audio Tools."""

import os
import tempfile
from unittest.mock import Mock, mock_open, patch

import pytest
from langchain_core.pydantic_v1 import SecretStr

from langchain_community.tools.audio import HuggingFaceTextToSpeechModelInference

AUDIO_FORMAT_EXT = "wav"


def test_huggingface_tts_constructor() -> None:
    with pytest.raises(ValueError):
        os.environ.pop("HUGGINGFACE_API_KEY", None)
        HuggingFaceTextToSpeechModelInference(
            model="test/model",
            file_extension=AUDIO_FORMAT_EXT,
        )

    with pytest.raises(ValueError):
        HuggingFaceTextToSpeechModelInference(
            model="test/model",
            file_extension=AUDIO_FORMAT_EXT,
            huggingface_api_key=SecretStr(""),
        )

    HuggingFaceTextToSpeechModelInference(
        model="test/model",
        file_extension=AUDIO_FORMAT_EXT,
        huggingface_api_key=SecretStr("foo"),
    )

    os.environ["HUGGINGFACE_API_KEY"] = "foo"
    HuggingFaceTextToSpeechModelInference(
        model="test/model",
        file_extension=AUDIO_FORMAT_EXT,
    )


def test_huggingface_tts_run_with_requests_mock() -> None:
    os.environ["HUGGINGFACE_API_KEY"] = "foo"

    with tempfile.TemporaryDirectory() as tmp_dir, patch(
        "requests.post"
    ) as mock_inference, patch("builtins.open", mock_open()) as mock_file:
        input_query = "Dummy input"

        expected_output_base_name = os.path.join(tmp_dir, "test_output")
        expected_output_path = f"{expected_output_base_name}.{AUDIO_FORMAT_EXT}"

        test_audio_content = b"test_audio_bytes"

        tts = HuggingFaceTextToSpeechModelInference(
            model="test/model",
            file_extension=AUDIO_FORMAT_EXT,
        )

        # Mock the requests.post response
        mock_response = Mock()
        mock_response.content = test_audio_content
        mock_inference.return_value = mock_response

        output_path = tts._run(
            query=input_query,
            output_base_name=expected_output_base_name,
        )

        assert output_path == expected_output_path

        mock_inference.assert_called_once_with(
            tts.api_url,
            headers={
                "Authorization": f"Bearer {tts.huggingface_api_key.get_secret_value()}"
            },
            json={"inputs": input_query},
        )

        mock_file.assert_called_once_with(expected_output_path, mode="xb")
        mock_file.return_value.write.assert_called_once_with(test_audio_content)
