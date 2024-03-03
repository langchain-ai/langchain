"""Test OpenAI Text-to-Speech API wrapper."""

import os
from unittest.mock import Mock, patch

import pytest

from langchain_openai import OpenAITextToSpeechTool

os.environ["OPENAI_API_KEY"] = "foo"


def test_openai_tts_invalid_speed() -> None:
    with pytest.raises(ValueError):
        OpenAITextToSpeechTool(speed=0)
    with pytest.raises(ValueError):
        OpenAITextToSpeechTool(speed=5)
    with pytest.raises(ValueError):
        OpenAITextToSpeechTool(speed=-1)


def test_openai_tts_run() -> None:
    tts = OpenAITextToSpeechTool()
    input = "Dumby input"
    output_dir, output_name = "test", "test_output"
    expected_output_path = f"{output_dir}/{output_name}.mp3"

    with patch("openai.audio.speech.create") as mock_tts:
        mock_tts.return_value = Mock()

        output_path = tts._run(
            input=input,
            output_dir=output_dir,
            output_name=output_name,
        )

        assert output_path == expected_output_path

        mock_tts.assert_called_once_with(
            model=tts.model.value,
            voice=tts.voice.value,
            speed=tts.speed,
            response_format=tts.format.value,
            input=input,
        )

        mock_tts.return_value.stream_to_file.assert_called_once_with(
            expected_output_path
        )
