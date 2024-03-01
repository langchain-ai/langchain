from __future__ import annotations

from typing import Any, Iterator, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.audio import AzureAISpeechParser

SPEECH_SERVICE_REGION = "eastasia"
SPEECH_SERVICE_KEY = "someservicekey"


# Loader for testing purposes only
class _AzureAISpeechLoader(BaseLoader):
    """Azure AI Speech Service Document Loader.
    A document loader that can load an audio file from the local file system
    and transcribe it using Azure AI Speech Service.


    Examples:
        .. code-block:: python
            from langchain_community.document_loaders import AzureAISpeechLoader
            loader = AzureAISpeechParser(
                file_path="path/to/directory/example.wav",
                api_key="speech-api-key-from-azure",
                region="speech-api-region-from-azure"
            )
            loader.lazy_load()
    """

    def __init__(self, file_path: str, **kwargs: Any) -> None:
        """
        Args:
            file_path: The path to the audio file.
        """
        self.file_path = file_path
        self.parser = AzureAISpeechParser(**kwargs)  # type: ignore

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        blob = Blob.from_path(self.file_path)
        return self.parser.lazy_parse(blob)


def _get_audio_file_path() -> str:
    return "../test_audio/whatstheweatherlike.wav"


def test_azure_speech_load_key_region_auto_detect_languages() -> None:
    loader = _AzureAISpeechLoader(
        _get_audio_file_path(),
        api_key=SPEECH_SERVICE_KEY,
        region=SPEECH_SERVICE_REGION,
        auto_detect_languages=["zh-CN", "en-US"],
    )
    documents = loader.load()
    assert "what" in documents[0].page_content.lower()


def test_azure_speech_load_key_region_language() -> None:
    loader = _AzureAISpeechLoader(
        _get_audio_file_path(),
        api_key=SPEECH_SERVICE_KEY,
        region=SPEECH_SERVICE_REGION,
        speech_recognition_language="en-US",
    )
    documents = loader.load()
    assert "what" in documents[0].page_content.lower()


def test_azure_speech_load_key_region() -> None:
    loader = _AzureAISpeechLoader(
        _get_audio_file_path(), api_key=SPEECH_SERVICE_KEY, region=SPEECH_SERVICE_REGION
    )
    documents = loader.load()
    assert "what" in documents[0].page_content.lower()


def test_azure_speech_load_key_endpoint() -> None:
    loader = _AzureAISpeechLoader(
        _get_audio_file_path(),
        api_key=SPEECH_SERVICE_KEY,
        endpoint=f"wss://{SPEECH_SERVICE_REGION}.stt.speech.microsoft.com/speech/recognition"
        "/conversation/cognitiveservices/v1",
    )
    documents = loader.load()
    assert "what" in documents[0].page_content.lower()
