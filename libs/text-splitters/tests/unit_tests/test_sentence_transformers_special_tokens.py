"""Regression tests for SentenceTransformersTokenTextSplitter special-token handling.

Issue #41624: `split_text()` stripped `encoded[1:-1]`, assuming every tokenizer
adds exactly one special token at each end (BOS+EOS). Tokenizers without them
(e.g. Qwen embedding models) lose content tokens: a single-token word vanishes.
The fix encodes with `add_special_tokens=False` instead of positional slicing.

Tests use a fake tokenizer via `sys.modules` patching — no heavy packages.
Importing the class at module level is safe: only `__init__` (called inside
fixtures) touches the `sentence_transformers` package.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from langchain_text_splitters import SentenceTransformersTokenTextSplitter

_WORDS: dict[str, int] = {"Hello": 11, "world": 12, "!": 13, "alpha": 14, "beta": 15}


class FakeTokenizer:
    """Fake HF-style tokenizer with configurable special-token behaviour."""

    def __init__(self, *, with_special: bool, bos: int = 0, eos: int = 1) -> None:
        self.with_special = with_special
        self.bos = bos
        self.eos = eos

    def encode(self, text: str, **kwargs: object) -> list[int]:
        ids = [_WORDS[w] for w in text.split() if w in _WORDS]
        if kwargs.get("add_special_tokens", True) and self.with_special:
            return [self.bos, *ids, self.eos]
        return ids

    def decode(self, ids: list[int]) -> str:
        reverse = {v: k for k, v in _WORDS.items()}
        return " ".join(reverse[i] for i in ids if i in reverse)


def _fake_sentence_transformers_module(*, with_special: bool) -> SimpleNamespace:
    def make_model(_name: str, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            tokenizer=FakeTokenizer(with_special=with_special),
            max_seq_length=512,
        )

    return SimpleNamespace(SentenceTransformer=make_model)


@pytest.fixture
def qwen_like_splitter(
    monkeypatch: pytest.MonkeyPatch,
) -> SentenceTransformersTokenTextSplitter:
    """Tokenizer WITHOUT BOS+EOS (Qwen-style): encode returns content tokens only."""
    fake = _fake_sentence_transformers_module(with_special=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    return SentenceTransformersTokenTextSplitter(
        model_name="fake/qwen-like", tokens_per_chunk=20, chunk_overlap=0
    )


@pytest.fixture
def bos_eos_splitter(
    monkeypatch: pytest.MonkeyPatch,
) -> SentenceTransformersTokenTextSplitter:
    """Tokenizer WITH BOS+EOS: regression guard — fix must not change behaviour."""
    fake = _fake_sentence_transformers_module(with_special=True)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    return SentenceTransformersTokenTextSplitter(
        model_name="fake/bos-eos", tokens_per_chunk=20, chunk_overlap=0
    )


class TestNoSpecialTokens:
    """Issue #41624: tokenizers without BOS+EOS must keep every content token."""

    def test_single_token_word_survives(
        self, qwen_like_splitter: SentenceTransformersTokenTextSplitter
    ) -> None:
        assert qwen_like_splitter.split_text("Hello") == ["Hello"]

    def test_no_tokens_dropped(
        self, qwen_like_splitter: SentenceTransformersTokenTextSplitter
    ) -> None:
        assert qwen_like_splitter.split_text("Hello world !") == ["Hello world !"]

    def test_multi_word_input(
        self, qwen_like_splitter: SentenceTransformersTokenTextSplitter
    ) -> None:
        assert qwen_like_splitter.split_text("Hello alpha beta") == ["Hello alpha beta"]


class TestWithSpecialTokens:
    """Regression guard: BOS+EOS tokenizers behave exactly as before the fix."""

    def test_special_tokens_not_in_output(
        self, bos_eos_splitter: SentenceTransformersTokenTextSplitter
    ) -> None:
        assert bos_eos_splitter.split_text("Hello world") == ["Hello world"]

    def test_single_token_word_survives(
        self, bos_eos_splitter: SentenceTransformersTokenTextSplitter
    ) -> None:
        assert bos_eos_splitter.split_text("Hello") == ["Hello"]


class TestCountTokensUnchanged:
    """count_tokens() keeps its pre-fix behaviour (author-requested constraint)."""

    def test_counts_include_special_tokens(
        self, bos_eos_splitter: SentenceTransformersTokenTextSplitter
    ) -> None:
        # "Hello world" = 2 content tokens + BOS + EOS = 4, unchanged by the fix.
        assert bos_eos_splitter.count_tokens(text="Hello world") == 4
