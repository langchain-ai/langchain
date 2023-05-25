"""Test formatting functionality."""

from langchain.base_language import _get_token_ids_default_method


class TestTokenCountingWithGPT2Tokenizer:
    def test_tokenization(self) -> None:
        # Check that the tokenization is consistent with the GPT-2 tokenizer
        assert _get_token_ids_default_method("This is a test") == [1212, 318, 257, 1332]

    def test_empty_token(self) -> None:
        assert len(_get_token_ids_default_method("")) == 0

    def test_multiple_tokens(self) -> None:
        assert len(_get_token_ids_default_method("a b c")) == 3

    def test_special_tokens(self) -> None:
        # test for consistency when the default tokenizer is changed
        assert len(_get_token_ids_default_method("a:b_c d")) == 6
