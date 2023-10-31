"""Test Google Speech-to-Text document loader.

You need to create a Google Cloud project and enable the Speech-to-Text API to run the
integration tests.
Follow the instructions in the example notebook:
google_speech_to_text.ipynb
to set up the app and configure authentication.
"""

import pytest

from langchain.document_loaders.google_speech_to_text import GoogleSpeechToTextLoader


@pytest.mark.requires("google_api_core")
def test_initialization() -> None:
    loader = GoogleSpeechToTextLoader(
        project_id="test_project_id", file_path="./testfile.mp3"
    )
    assert loader.project_id == "test_project_id"
    assert loader.file_path == "./testfile.mp3"
    assert loader.location == "us-central1"
    assert loader.recognizer_id == "_"


@pytest.mark.requires("google.api_core")
def test_load() -> None:
    loader = GoogleSpeechToTextLoader(
        project_id="test_project_id", file_path="./testfile.mp3"
    )
    docs = loader.load()
    assert len(docs) == 1
    assert docs[0].page_content == "Test transcription text"
    assert docs[0].metadata["language_code"] == "en-US"
