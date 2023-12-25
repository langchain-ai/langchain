from langchain_community.document_loaders import AzureAISpeechLoader

SPEECH_SERVICE_REGION = "eastasia"
SPEECH_SERVICE_KEY = "c77dcf2aa5c04dd6b6613f77d9d9161d"


def _get_audio_file_path() -> str:
    return "../test_audio/whatstheweatherlike.wav"


def test_azure_speech_load_key_region_auto_detect_languages() -> None:
    loader = AzureAISpeechLoader(
        _get_audio_file_path(),
        api_key=SPEECH_SERVICE_KEY,
        region=SPEECH_SERVICE_REGION,
        auto_detect_languages=["zh-CN", "en-US"],
    )
    documents = loader.lazy_load()
    assert "what" in documents[0].page_content.lower()


def test_azure_speech_load_key_region_language() -> None:
    loader = AzureAISpeechLoader(
        _get_audio_file_path(),
        api_key=SPEECH_SERVICE_KEY,
        region=SPEECH_SERVICE_REGION,
        speech_recognition_language="en-US",
    )
    documents = loader.lazy_load()
    assert "what" in documents[0].page_content.lower()


def test_azure_speech_load_key_region() -> None:
    loader = AzureAISpeechLoader(
        _get_audio_file_path(),
        api_key=SPEECH_SERVICE_KEY,
        region=SPEECH_SERVICE_REGION
    )
    documents = loader.lazy_load()
    assert "what" in documents[0].page_content.lower()


def test_azure_speech_load_key_endpoint() -> None:
    loader = AzureAISpeechLoader(
        _get_audio_file_path(),
        api_key=SPEECH_SERVICE_KEY,
        endpoint=f"wss://{SPEECH_SERVICE_REGION}.stt.speech.microsoft.com/speech/recognition"
        "/conversation/cognitiveservices/v1",
    )
    documents = loader.lazy_load()
    assert "what" in documents[0].page_content.lower()
