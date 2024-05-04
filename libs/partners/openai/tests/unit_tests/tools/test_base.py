"""Test OpenAI Text-to-Speech API wrapper."""

import os
import tempfile
import uuid
from unittest.mock import Mock, mock_open, patch

from langchain_openai import OpenAITextToSpeechTool

os.environ["OPENAI_API_KEY"] = "foo"


def test_openai_tts_run() -> None:
    model, voice, file_extension = "tts-1", "alloy", "flac"
    with tempfile.TemporaryDirectory() as tmp_dir, patch(
        "uuid.uuid4"
    ) as mock_uuid, patch(
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
            speed=1.5,
            file_extension=file_extension,
        )

        with patch("openai.audio.speech.create") as mock_tts:
            mock_tts.return_value = Mock()
            mock_tts.content = test_audio_content

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
