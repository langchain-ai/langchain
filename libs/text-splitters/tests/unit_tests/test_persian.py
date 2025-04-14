"""Tests for Persian text processing components."""
import pytest

from langchain_text_splitters.persian import (
    PersianTokenizer,
    PersianTextNormalizer,
    PersianNumberConverter,
)


def test_basic_tokenization():
    """Test basic Persian text tokenization."""
    tokenizer = PersianTokenizer()
    text = "سلام دنیا"
    tokens = tokenizer.tokenize(text)
    assert tokens == ["سلام", "دنیا"]


def test_mixed_text_tokenization():
    """Test tokenization of mixed Persian-English text."""
    tokenizer = PersianTokenizer()
    text = "این یک text ترکیبی است"
    tokens = tokenizer.tokenize(text)
    assert tokens == ["این", "یک", "text", "ترکیبی", "است"]


def test_zwnj_handling():
    """Test handling of ZWNJ character in tokenization."""
    tokenizer = PersianTokenizer()
    text = "می‌خواهم"  # Text with ZWNJ
    tokens = tokenizer.tokenize(text)
    assert tokens == ["می‌خواهم"]


def test_character_normalization():
    """Test Persian character normalization."""
    normalizer = PersianTextNormalizer()
    text = "كتاب"  # With Arabic kaf
    normalized = normalizer.normalize(text)
    assert normalized == "کتاب"  # With Persian kaf


def test_diacritic_removal():
    """Test removal of diacritical marks."""
    normalizer = PersianTextNormalizer()
    text = "کِتاب"  # With kasra
    normalized = normalizer.normalize(text)
    assert normalized == "کتاب"


def test_space_normalization():
    """Test normalization of spaces in text."""
    normalizer = PersianTextNormalizer()
    text = "این  متن    دارای    فاصله    است"
    normalized = normalizer.normalize(text)
    assert normalized == "این متن دارای فاصله است"


def test_persian_to_english_numbers():
    """Test conversion of Persian numbers to English."""
    converter = PersianNumberConverter()
    text = "۱۲۳۴۵"
    converted = converter.to_english(text)
    assert converted == "12345"


def test_english_to_persian_numbers():
    """Test conversion of English numbers to Persian."""
    converter = PersianNumberConverter()
    text = "12345"
    converted = converter.to_persian(text)
    assert converted == "۱۲۳۴۵"


def test_mixed_numbers():
    """Test handling of mixed number systems in text."""
    converter = PersianNumberConverter()
    text = "The number ۱۲۳ is equal to 123"
    converted = converter.to_persian(text)
    assert converted == "The number ۱۲۳ is equal to ۱۲۳"


def test_arabic_numbers():
    """Test handling of Arabic numbers."""
    converter = PersianNumberConverter()
    text = "١٢٣٤٥"
    converted = converter.to_english(text)
    assert converted == "12345"


def test_tokenizer_empty_text():
    """Test tokenizer behavior with empty text."""
    tokenizer = PersianTokenizer()
    assert tokenizer.tokenize("") == []


def test_normalizer_empty_text():
    """Test normalizer behavior with empty text."""
    normalizer = PersianTextNormalizer()
    assert normalizer.normalize("") == ""


def test_number_converter_empty_text():
    """Test number converter behavior with empty text."""
    converter = PersianNumberConverter()
    assert converter.to_persian("") == ""
    assert converter.to_english("") == "" 