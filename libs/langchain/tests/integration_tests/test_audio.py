from pathlib import Path

from langchain.document_loaders.audio import AzureSpeechServiceLoader

SPEECH_SERVICE_REGION = ""
SPEECH_SERVICE_KEY = ""


def _get_csv_file_path() -> str:
    return str(
        Path(__file__).resolve().parent.parent
        / "test_audio"
        / "whatstheweatherlike.wav"
    )


def test_azure_speech_load_key_region_auto_detect_languages() -> None:
    loader = AzureSpeechServiceLoader.from_path(
        _get_csv_file_path(),
        key=SPEECH_SERVICE_KEY,
        region=SPEECH_SERVICE_REGION,
        auto_detect_languages=["zh-CN", "en-US"],
    )
    documents = loader.load()
    assert "what" in documents[0].page_content.lower()


def test_azure_speech_load_key_region_language() -> None:
    loader = AzureSpeechServiceLoader.from_path(
        _get_csv_file_path(),
        key=SPEECH_SERVICE_KEY,
        region=SPEECH_SERVICE_REGION,
        language="en-US",
    )
    documents = loader.load()
    assert "what" in documents[0].page_content.lower()


def test_azure_speech_load_key_region() -> None:
    loader = AzureSpeechServiceLoader.from_path(
        _get_csv_file_path(), key=SPEECH_SERVICE_KEY, region=SPEECH_SERVICE_REGION
    )
    documents = loader.load()
    assert "what" in documents[0].page_content.lower()


def test_azure_speech_load_key_endpoint() -> None:
    loader = AzureSpeechServiceLoader.from_path(
        _get_csv_file_path(),
        key=SPEECH_SERVICE_KEY,
        endpoint=(
            f"wss://{SPEECH_SERVICE_REGION}.stt.speech.microsoft.com/speech/recognition"
            "/conversation/cognitiveservices/v1",
        ),
    )
    documents = loader.load()
    assert "what" in documents[0].page_content.lower()
