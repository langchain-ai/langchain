"""Test text splitting functionality."""
import pytest

from langchain.text_splitter import (
    CharacterTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
)


def test_character_text_splitter() -> None:
    """Test splitting by character count."""
    text = "foo bar baz 123"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=5, chunk_overlap=3)
    output = splitter.split_text(text)
    expected_output = ["foo bar", "bar baz", "baz 123"]
    assert output == expected_output


def test_character_text_splitter_longer_words() -> None:
    """Test splitting by characters when splits not found easily."""
    text = "foo bar baz 123"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1, chunk_overlap=1)
    output = splitter.split_text(text)
    expected_output = ["foo", "bar", "baz", "123"]
    assert output == expected_output


def test_character_text_splitting_args() -> None:
    """Test invalid arguments."""
    with pytest.raises(ValueError):
        CharacterTextSplitter(chunk_size=2, chunk_overlap=4)


def test_nltk_text_splitting_args() -> None:
    """Test invalid arguments."""
    with pytest.raises(ValueError):
        NLTKTextSplitter(chunk_size=2, chunk_overlap=4)


def test_spacy_text_splitting_args() -> None:
    """Test invalid arguments."""
    with pytest.raises(ValueError):
        SpacyTextSplitter(chunk_size=2, chunk_overlap=4)


def test_nltk_text_splitter() -> None:
    """Test splitting by sentence using NLTK."""
    text = "This is sentence one. And this is sentence two."
    separator = "|||"
    splitter = NLTKTextSplitter(separator=separator)
    output = splitter.split_text(text)
    expected_output = [f"This is sentence one.{separator}And this is sentence two."]
    assert output == expected_output


def test_spacy_text_splitter() -> None:
    """Test splitting by sentence using Spacy.

    Note: First run of this test will download the spacy model, and it can be slow.
    """
    text = "This is sentence one. And this is sentence two."
    separator = "|||"
    splitter = SpacyTextSplitter(separator=separator)
    output = splitter.split_text(text)
    expected_output = [f"This is sentence one.{separator}And this is sentence two."]
    assert output == expected_output
