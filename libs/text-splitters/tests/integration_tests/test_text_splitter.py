"""Test text splitters that require an integration."""

import pytest

from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters.character import CharacterTextSplitter


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


def test_token_text_splitter() -> None:
    """Test no overlap."""
    splitter = TokenTextSplitter(chunk_size=5, chunk_overlap=0)
    output = splitter.split_text("abcdef" * 5)  # 10 token string
    expected_output = ["abcdefabcdefabc", "defabcdefabcdef"]
    assert output == expected_output


def test_token_text_splitter_overlap() -> None:
    """Test with overlap."""
    splitter = TokenTextSplitter(chunk_size=5, chunk_overlap=1)
    output = splitter.split_text("abcdef" * 5)  # 10 token string
    expected_output = ["abcdefabcdefabc", "abcdefabcdefabc", "abcdef"]
    assert output == expected_output


def test_token_text_splitter_from_tiktoken() -> None:
    splitter = TokenTextSplitter.from_tiktoken_encoder(model_name="gpt-3.5-turbo")
    expected_tokenizer = "cl100k_base"
    actual_tokenizer = splitter._tokenizer.name
    assert expected_tokenizer == actual_tokenizer
