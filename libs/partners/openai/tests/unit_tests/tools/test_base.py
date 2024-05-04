"""Test OpenAI Text-to-Speech API wrapper."""

import os
import tempfile
import uuid
from typing import Literal
from unittest.mock import Mock, mock_open, patch

from langchain_openai import OpenAITextToSpeechTool

os.environ["OPENAI_API_KEY"] = "foo"


def test_openai_tts_run() -> None:
    model = "tts-1"
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy"
    file_extension: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "wav"

    with tempfile.TemporaryDirectory() as tmp_dir, patch(
        "uuid.uuid4"
    ) as mock_uuid, patch("openai.audio.speech.create") as mock_tts, patch(
        "builtins.open", mock_open()
    ) as mock_file:
        input_query = "Dummy input"

        mock_uuid_value = uuid.UUID("00000000-0000-0000-0000-000000000000")
        mock_uuid.return_value = mock_uuid_value

        expected_output_file_base_name = os.path.join(tmp_dir, str(mock_uuid_value))
        expected_output_file = f"{expected_output_file_base_name}.{file_extension}"

        test_audio_content = b"test_audio_bytes"

        tts = OpenAITextToSpeechTool(
            model=model,
            voice=voice,
            file_extension=file_extension,
            speed=1.5,
            destination_dir=tmp_dir,
            file_naming_func="uuid",
        )

        # Mock the OpenAI response
        mock_response = Mock()
        mock_response.content = test_audio_content
        mock_tts.return_value = mock_response

        output_path = tts._run(query=input_query)

        assert output_path == expected_output_file

        mock_tts.assert_called_once_with(
            model=tts.model,
            voice=tts.voice,
            speed=tts.speed,
            response_format=file_extension,
            input=input_query,
        )

        mock_file.assert_called_once_with(expected_output_file, mode="xb")
        mock_file.return_value.write.assert_called_once_with(test_audio_content)
