from typing import List

import pytest


@pytest.mark.requires("langdetect")
@pytest.mark.parametrize(
    "text,language",
    [
        ("Hello, my name is John Doe.", "en"),
        ("Hallo, mein Name ist John Doe.", "de"),
    ],
)
def test_detect_single_language(text: str, language: str) -> str:
    """Test detecting most probable language of a text"""
    from langchain_experimental.language_detector import LangDetector

    lang_detector = LangDetector()
    predicted = lang_detector.detect_single_language(text)
    assert predicted == language


@pytest.mark.requires("langdetect")
@pytest.mark.parametrize(
    "text,languages",
    [
        ("Hello, my name is John Doe.", ["en"]),
        ("Hello, my name is John Doe. I live in London. Auf Wiedersehen.", ["de", "en"]),
    ],
)
def test_detect_many_languages(text: str, languages: List[str]) -> str:
    """Test detecting most probable languages of a text"""
    from langchain_experimental.language_detector import LangDetector

    lang_detector = LangDetector()
    predicted = lang_detector.detect_many_languages(text)
    assert [x[0] for x in predicted] == languages
