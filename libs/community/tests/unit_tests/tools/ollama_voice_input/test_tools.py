"""Tests for VoiceInputChain & SpeechToText module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.documents.base import Blob

from langchain_community.tools.ollama_voice_input import SpeechToText, VoiceInputChain

# Replace `module_name` with the actual module name

_THIS_DIR = Path(__file__).parents[3]

_EXAMPLES_DIR = _THIS_DIR / "examples"
AUDIO_M4A = _EXAMPLES_DIR / "hello_world.m4a"
LONG_AUDIO = _EXAMPLES_DIR / "long_audio.wav"


# test no audio
def test_no_audio() -> None:
    """Test that an error is raised when no audio input is provided."""
    with pytest.raises(ValueError):
        parser = SpeechToText()
        SpeechToText().lazy_parse(parser.audio_blob)


# test short audio
@pytest.mark.requires("openai")
def test_short_audio_transcription() -> None:
    """Test lazy parsing of short audio files using SpeechToText."""

    # Env variable should be set to avoid OpenAI API key error
    parser = SpeechToText(api_key="key", audio_path=AUDIO_M4A)
    result = list(parser.lazy_parse(parser.audio_blob))

    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].page_content == "Hello world!"


# test long audio
@pytest.mark.requires("openai")
def test_long_audio_transcription() -> None:
    """Test lazy parsing of long audio files using SpeechToText."""

    # Env variable should be set to avoid OpenAI API key error
    parser = SpeechToText(api_key="key", audio_path=LONG_AUDIO)
    result = list(parser.lazy_parse(parser.audio_blob))

    assert len(result) > 1
    assert all(isinstance(doc, Document) for doc in result)


# test VoiceInputChain run on audio input (short)
@pytest.mark.requires("openai")
@patch("langchain_community.llms.ollama.Ollama.invoke")
@patch("langchain_community.document_loaders.generic.GenericLoader.load")
def test_voice_input_chain_run(mock_loader: Mock, mock_invoke: Mock) -> None:
    """Test VoiceInputChain.run for short input that fits within the context window."""

    # Mock functions responses
    mock_loader.return_value = [Document(page_content="Short input text")]
    mock_invoke.return_value = "Processed short input text"

    # Setup mocked SpeechToText instance
    mock_stt = Mock(spec=SpeechToText)
    mock_stt.audio_blob = Mock(spec=Blob)
    mock_stt.audio_blob.path = AUDIO_M4A

    # Initialize the VoiceInputChain
    voice_chain = VoiceInputChain(stt=mock_stt, model="llama2")
    response = voice_chain.run()

    mock_invoke.assert_called_once_with("Short input text")
    assert response == "Processed short input text"
