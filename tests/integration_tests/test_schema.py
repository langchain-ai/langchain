"""Test formatting functionality."""

from langchain.base_language import _get_num_tokens_default_method


class TestTokenCountingWithGPT2Tokenizer:
    def test_empty_token(self) -> None:
        assert _get_num_tokens_default_method("") == 0

    def test_multiple_tokens(self) -> None:
        assert _get_num_tokens_default_method("a b c") == 3

    def test_special_tokens(self) -> None:
        # test for consistency when the default tokenizer is changed
        assert _get_num_tokens_default_method("a:b_c d") == 6
