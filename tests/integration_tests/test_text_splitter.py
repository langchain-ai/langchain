"""Test text splitters that require an integration."""

import pytest

from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter


def test_huggingface_type_check() -> None:
    """Test that type checks are done properly on input."""
    with pytest.raises(ValueError):
        CharacterTextSplitter.from_huggingface_tokenizer("foo")


def test_huggingface_tokenizer() -> None:
    """Test text splitter that uses a HuggingFace tokenizer."""
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, separator=" ", chunk_size=1, chunk_overlap=0
    )
    output = text_splitter.split_text("foo bar")
    assert output == ["foo", "bar"]


class TestTokenTextSplitter:
    """Test token text splitter."""

    def test_basic(self) -> None:
        """Test no overlap."""
        splitter = TokenTextSplitter(chunk_size=5, chunk_overlap=0)
        output = splitter.split_text("abcdef" * 5)  # 10 token string
        expected_output = ["abcdefabcdefabc", "defabcdefabcdef"]
        assert output == expected_output

    def test_overlap(self) -> None:
        """Test with overlap."""
        splitter = TokenTextSplitter(chunk_size=5, chunk_overlap=1)
        output = splitter.split_text("abcdef" * 5)  # 10 token string
        expected_output = ["abcdefabcdefabc", "abcdefabcdefabc", "abcdef"]
        assert output == expected_output
