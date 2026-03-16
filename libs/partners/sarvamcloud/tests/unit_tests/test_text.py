"""Unit tests for Sarvam AI text services."""

from unittest.mock import MagicMock, patch

import pytest


class TestSarvamTranslator:
    @pytest.fixture()
    def translator(self) -> "SarvamTranslator":
        from langchain_sarvamcloud.text import SarvamTranslator

        mock_client = MagicMock()
        mock_client.text.translate.return_value = {
            "request_id": "req-tr-1",
            "translated_text": "नमस्ते, आप कैसे हैं?",
            "source_language_code": "en-IN",
        }
        with patch("sarvamai.SarvamAI", return_value=mock_client):
            tr = SarvamTranslator(
                model="sarvam-translate:v1",
                api_subscription_key="test-key",  # type: ignore[arg-type]
            )
        tr._client = mock_client
        return tr

    def test_translate_returns_translated_text(
        self, translator: "SarvamTranslator"
    ) -> None:
        result = translator.translate(
            "Hello, how are you?",
            source_language_code="en-IN",
            target_language_code="hi-IN",
        )
        assert "translated_text" in result
        assert result["translated_text"] == "नमस्ते, आप कैसे हैं?"

    def test_translate_passes_mode(self, translator: "SarvamTranslator") -> None:
        translator.translate(
            "Hello",
            source_language_code="en-IN",
            target_language_code="hi-IN",
            mode="code-mixed",
        )
        call_kwargs = translator._client.text.translate.call_args[1]
        assert call_kwargs["mode"] == "code-mixed"

    def test_translate_passes_speaker_gender(
        self, translator: "SarvamTranslator"
    ) -> None:
        translator.translate(
            "Hello",
            source_language_code="en-IN",
            target_language_code="hi-IN",
            speaker_gender="Female",
        )
        call_kwargs = translator._client.text.translate.call_args[1]
        assert call_kwargs["speaker_gender"] == "Female"

    def test_translate_passes_output_script(
        self, translator: "SarvamTranslator"
    ) -> None:
        translator.translate(
            "Hello",
            source_language_code="en-IN",
            target_language_code="hi-IN",
            output_script="roman",
        )
        call_kwargs = translator._client.text.translate.call_args[1]
        assert call_kwargs["output_script"] == "roman"

    def test_default_mode_is_formal(self) -> None:
        from langchain_sarvamcloud.text import SarvamTranslator

        with patch("sarvamai.SarvamAI"):
            tr = SarvamTranslator(api_subscription_key="key")  # type: ignore[arg-type]
        assert tr.mode == "formal"

    def test_default_numerals_format_is_international(self) -> None:
        from langchain_sarvamcloud.text import SarvamTranslator

        with patch("sarvamai.SarvamAI"):
            tr = SarvamTranslator(api_subscription_key="key")  # type: ignore[arg-type]
        assert tr.numerals_format == "international"


class TestSarvamTransliterator:
    @pytest.fixture()
    def transliterator(self) -> "SarvamTransliterator":
        from langchain_sarvamcloud.text import SarvamTransliterator

        mock_client = MagicMock()
        mock_client.text.transliterate.return_value = {
            "request_id": "req-tl-1",
            "transliterated_text": "namaste",
            "source_language_code": "hi-IN",
        }
        with patch("sarvamai.SarvamAI", return_value=mock_client):
            tl = SarvamTransliterator(
                api_subscription_key="test-key"  # type: ignore[arg-type]
            )
        tl._client = mock_client
        return tl

    def test_transliterate_devanagari_to_roman(
        self, transliterator: "SarvamTransliterator"
    ) -> None:
        result = transliterator.transliterate(
            "नमस्ते",
            source_language_code="hi-IN",
            target_language_code="en-IN",
        )
        assert result["transliterated_text"] == "namaste"

    def test_indic_to_indic_raises_error(
        self, transliterator: "SarvamTransliterator"
    ) -> None:
        with pytest.raises(ValueError, match="Indic-to-Indic"):
            transliterator.transliterate(
                "नमस्ते",
                source_language_code="hi-IN",
                target_language_code="ta-IN",
            )

    def test_transliterate_passes_spoken_form(
        self, transliterator: "SarvamTransliterator"
    ) -> None:
        transliterator.transliterate(
            "नमस्ते",
            source_language_code="hi-IN",
            target_language_code="en-IN",
            spoken_form=True,
        )
        call_kwargs = transliterator._client.text.transliterate.call_args[1]
        assert call_kwargs["spoken_form"] is True

    def test_transliterate_passes_numerals_format(
        self, transliterator: "SarvamTransliterator"
    ) -> None:
        transliterator.transliterate(
            "नमस्ते 123",
            source_language_code="hi-IN",
            target_language_code="en-IN",
            numerals_format="native",
        )
        call_kwargs = transliterator._client.text.transliterate.call_args[1]
        assert call_kwargs["numerals_format"] == "native"


class TestSarvamLanguageDetector:
    @pytest.fixture()
    def detector(self) -> "SarvamLanguageDetector":
        from langchain_sarvamcloud.text import SarvamLanguageDetector

        mock_client = MagicMock()
        mock_client.text.identify_language.return_value = {
            "request_id": "req-lid-1",
            "language_code": "hi-IN",
            "script_code": "Deva",
            "confidence": 0.98,
        }
        with patch("sarvamai.SarvamAI", return_value=mock_client):
            det = SarvamLanguageDetector(
                api_subscription_key="test-key"  # type: ignore[arg-type]
            )
        det._client = mock_client
        return det

    def test_detect_returns_language_code(
        self, detector: "SarvamLanguageDetector"
    ) -> None:
        result = detector.detect("नमस्ते, आप कैसे हैं?")
        assert result["language_code"] == "hi-IN"

    def test_detect_returns_script_code(
        self, detector: "SarvamLanguageDetector"
    ) -> None:
        result = detector.detect("नमस्ते")
        assert result["script_code"] == "Deva"

    def test_detect_returns_confidence(
        self, detector: "SarvamLanguageDetector"
    ) -> None:
        result = detector.detect("नमस्ते")
        assert result["confidence"] == pytest.approx(0.98)

    def test_detect_calls_identify_language(
        self, detector: "SarvamLanguageDetector"
    ) -> None:
        detector.detect("test text")
        detector._client.text.identify_language.assert_called_once_with(
            input="test text"
        )
