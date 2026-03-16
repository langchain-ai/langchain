"""Test that all public classes can be imported."""


def test_chat_sarvam_import() -> None:
    from langchain_sarvamcloud import ChatSarvam  # noqa: F401


def test_sarvam_stt_import() -> None:
    from langchain_sarvamcloud import SarvamSTT  # noqa: F401


def test_sarvam_batch_stt_import() -> None:
    from langchain_sarvamcloud import SarvamBatchSTT  # noqa: F401


def test_sarvam_tts_import() -> None:
    from langchain_sarvamcloud import SarvamTTS  # noqa: F401


def test_sarvam_translator_import() -> None:
    from langchain_sarvamcloud import SarvamTranslator  # noqa: F401


def test_sarvam_transliterator_import() -> None:
    from langchain_sarvamcloud import SarvamTransliterator  # noqa: F401


def test_sarvam_language_detector_import() -> None:
    from langchain_sarvamcloud import SarvamLanguageDetector  # noqa: F401


def test_sarvam_document_intelligence_import() -> None:
    from langchain_sarvamcloud import SarvamDocumentIntelligence  # noqa: F401


def test_version_import() -> None:
    from langchain_sarvamcloud import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0
