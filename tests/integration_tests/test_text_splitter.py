"""Test text splitters that require an integration."""

import pytest

from langchain.text_splitter import (
    CharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TokenTextSplitter,
)


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


def test_sentence_transformers_count_tokens() -> None:
    splitter = SentenceTransformersTokenTextSplitter(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    text = "Lorem ipsum"

    token_count = splitter.count_tokens(text=text)

    expected_start_stop_token_count = 2
    expected_text_token_count = 2
    expected_token_count = expected_start_stop_token_count + expected_text_token_count

    assert expected_token_count == token_count


def test_sentence_transformers_split_text() -> None:
    splitter = SentenceTransformersTokenTextSplitter(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    text = "Lorem ipsum"
    text_chunks = splitter.split_text(text=text)
    expected_text_chunks = [text]
    assert expected_text_chunks == text_chunks


def test_sentence_transformers_multiple_tokens() -> None:
    splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0)
    text = "Lorem "

    count_start_and_end_tokens = 2
    text_token_count = splitter.count_tokens(text=text) - count_start_and_end_tokens
    token_multiplier = splitter.maximum_tokens_per_chunk // text_token_count + 1
    text_chunks = splitter.split_text(text=text * token_multiplier)

    expected_number_of_chunks = 2

    assert expected_number_of_chunks == len(text_chunks)
    actual = splitter.count_tokens(text=text_chunks[1]) - count_start_and_end_tokens
    expected = token_multiplier * text_token_count - splitter.maximum_tokens_per_chunk
    assert expected == actual
