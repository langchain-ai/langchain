"""Test Audio Tools."""

import os
import tempfile
import uuid
from unittest.mock import Mock, mock_open, patch

import pytest
from pydantic import SecretStr

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

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch("uuid.uuid4") as mock_uuid,
        patch("requests.post") as mock_inference,
        patch("builtins.open", mock_open()) as mock_file,
    ):
        input_query = "Dummy input"

        mock_uuid_value = uuid.UUID("00000000-0000-0000-0000-000000000000")
        mock_uuid.return_value = mock_uuid_value

        expected_output_file_base_name = os.path.join(tmp_dir, str(mock_uuid_value))
        expected_output_file = f"{expected_output_file_base_name}.{AUDIO_FORMAT_EXT}"

        test_audio_content = b"test_audio_bytes"

        tts = HuggingFaceTextToSpeechModelInference(
            model="test/model",
            file_extension=AUDIO_FORMAT_EXT,
            destination_dir=tmp_dir,
            file_naming_func="uuid",
        )

        # Mock the requests.post response
        mock_response = Mock()
        mock_response.content = test_audio_content
        mock_inference.return_value = mock_response

        output_path = tts._run(input_query)

        assert output_path == expected_output_file

        mock_inference.assert_called_once_with(
            tts.api_url,
            headers={
                "Authorization": f"Bearer {tts.huggingface_api_key.get_secret_value()}"
            },
            json={"inputs": input_query},
        )

        mock_file.assert_called_once_with(expected_output_file, mode="xb")
        mock_file.return_value.write.assert_called_once_with(test_audio_content)
