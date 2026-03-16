"""Integration tests for text services — requires SARVAM_API_SUBSCRIPTION_KEY."""

import pytest

from langchain_sarvamcloud.text import (
    SarvamLanguageDetector,
    SarvamTransliterator,
    SarvamTranslator,
)


@pytest.mark.requires("SARVAM_API_SUBSCRIPTION_KEY")
class TestSarvamTranslatorIntegration:
    def test_translate_english_to_hindi(self) -> None:
        translator = SarvamTranslator(model="sarvam-translate:v1")
        result = translator.translate(
            "Hello, how are you?",
            source_language_code="en-IN",
            target_language_code="hi-IN",
        )
        assert "translated_text" in result
        assert len(result["translated_text"]) > 0

    def test_translate_with_code_mixed_mode(self) -> None:
        translator = SarvamTranslator(model="mayura:v1")
        result = translator.translate(
            "I am going to office.",
            source_language_code="en-IN",
            target_language_code="hi-IN",
            mode="code-mixed",
        )
        assert "translated_text" in result


@pytest.mark.requires("SARVAM_API_SUBSCRIPTION_KEY")
class TestSarvamTransliteratorIntegration:
    def test_devanagari_to_roman(self) -> None:
        tl = SarvamTransliterator()
        result = tl.transliterate(
            "नमस्ते",
            source_language_code="hi-IN",
            target_language_code="en-IN",
        )
        assert "transliterated_text" in result
        assert len(result["transliterated_text"]) > 0

    def test_roman_to_devanagari(self) -> None:
        tl = SarvamTransliterator()
        result = tl.transliterate(
            "namaste",
            source_language_code="en-IN",
            target_language_code="hi-IN",
        )
        assert "transliterated_text" in result


@pytest.mark.requires("SARVAM_API_SUBSCRIPTION_KEY")
class TestSarvamLanguageDetectorIntegration:
    def test_detect_hindi(self) -> None:
        detector = SarvamLanguageDetector()
        result = detector.detect("नमस्ते, आप कैसे हैं?")
        assert result.get("language_code") == "hi-IN"

    def test_detect_english(self) -> None:
        detector = SarvamLanguageDetector()
        result = detector.detect("Hello, how are you doing today?")
        assert result.get("language_code") == "en-IN"

    def test_detect_returns_confidence(self) -> None:
        detector = SarvamLanguageDetector()
        result = detector.detect("नमस्ते")
        confidence = result.get("confidence")
        if confidence is not None:
            assert 0.0 <= confidence <= 1.0
