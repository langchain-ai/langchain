"""Test text splitting functionality."""
import pytest

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter


def test_character_text_splitter() -> None:
    """Test splitting by character count."""
    text = "foo bar baz 123"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=7, chunk_overlap=3)
    output = splitter.split_text(text)
    expected_output = ["foo bar", "bar baz", "baz 123"]
    assert output == expected_output


def test_character_text_splitter_long() -> None:
    """Test splitting by character count on long words."""
    text = "foo bar baz a a"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=1)
    output = splitter.split_text(text)
    expected_output = ["foo", "bar", "baz", "a a"]
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


def test_create_documents() -> None:
    """Test create documents method."""
    texts = ["foo bar", "baz"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    docs = splitter.create_documents(texts)
    expected_docs = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    assert docs == expected_docs


def test_create_documents_with_metadata() -> None:
    """Test create documents with metadata method."""
    texts = ["foo bar", "baz"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    docs = splitter.create_documents(texts, [{"source": "1"}, {"source": "2"}])
    expected_docs = [
        Document(page_content="foo", metadata={"source": "1"}),
        Document(page_content="bar", metadata={"source": "1"}),
        Document(page_content="baz", metadata={"source": "2"}),
    ]
    assert docs == expected_docs
