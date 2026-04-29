"""Unit tests for `NLTKTextSplitter`."""

from __future__ import annotations

import pytest

from langchain_text_splitters import nltk as nltk_module
from langchain_text_splitters.nltk import NLTKTextSplitter


def test_missing_nltk_raises_import_error_before_arg_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing NLTK install must surface before argument validation.

    `ImportError` for a missing NLTK install must take precedence over
    argument-combination validation, so users see the real prerequisite first.
    """
    monkeypatch.setattr(nltk_module, "_HAS_NLTK", False)

    with pytest.raises(ImportError, match="NLTK is not installed"):
        NLTKTextSplitter(use_span_tokenize=True)


def test_missing_nltk_raises_import_error_with_default_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing NLTK install raises `ImportError` even with default args.

    Even with valid argument combinations, a missing NLTK install must
    raise `ImportError`.
    """
    monkeypatch.setattr(nltk_module, "_HAS_NLTK", False)

    with pytest.raises(ImportError, match="NLTK is not installed"):
        NLTKTextSplitter()
