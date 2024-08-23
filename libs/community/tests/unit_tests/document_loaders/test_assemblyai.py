import pytest
import responses
from pytest_mock import MockerFixture
from requests import HTTPError

from langchain_community.document_loaders import (
    AssemblyAIAudioLoaderById,
    AssemblyAIAudioTranscriptLoader,
)
from langchain_community.document_loaders.assemblyai import TranscriptFormat


@pytest.mark.requires("assemblyai")
def test_initialization() -> None:
    loader = AssemblyAIAudioTranscriptLoader(
        file_path="./testfile.mp3", api_key="api_key"
    )
    assert loader.file_path == "./testfile.mp3"
    assert loader.transcript_format == TranscriptFormat.TEXT


@pytest.mark.requires("assemblyai")
def test_load(mocker: MockerFixture) -> None:
    mocker.patch(
        "assemblyai.Transcriber.transcribe",
        return_value=mocker.MagicMock(
            text="Test transcription text", json_response={"id": "1"}, error=None
        ),
    )

    loader = AssemblyAIAudioTranscriptLoader(
        file_path="./testfile.mp3", api_key="api_key"
    )
    docs = loader.load()
    assert len(docs) == 1
    assert docs[0].page_content == "Test transcription text"
    assert docs[0].metadata == {"id": "1"}


@pytest.mark.requires("assemblyai")
def test_transcription_error(mocker: MockerFixture) -> None:
    mocker.patch(
        "assemblyai.Transcriber.transcribe",
        return_value=mocker.MagicMock(error="Test error"),
    )

    loader = AssemblyAIAudioTranscriptLoader(
        file_path="./testfile.mp3", api_key="api_key"
    )

    expected_error = "Could not transcribe file: Test error"
    with pytest.raises(ValueError, match=expected_error):
        loader.load()


@pytest.mark.requires("assemblyai")
@responses.activate
def test_load_by_id() -> None:
    responses.add(
        responses.GET,
        "https://api.assemblyai.com/v2/transcript/1234",
        json={"text": "Test transcription text", "id": "1234"},
        status=200,
    )

    loader = AssemblyAIAudioLoaderById(
        transcript_id="1234", api_key="api_key", transcript_format=TranscriptFormat.TEXT
    )
    docs = loader.load()
    assert len(docs) == 1
    assert docs[0].page_content == "Test transcription text"
    assert docs[0].metadata == {"text": "Test transcription text", "id": "1234"}


@pytest.mark.requires("assemblyai")
@responses.activate
def test_transcription_error_by_id() -> None:
    responses.add(
        responses.GET,
        "https://api.assemblyai.com/v2/transcript/1234",
        status=404,
    )
    loader = AssemblyAIAudioLoaderById(
        transcript_id="1234", api_key="api_key", transcript_format=TranscriptFormat.TEXT
    )
    with pytest.raises(HTTPError):
        loader.load()
