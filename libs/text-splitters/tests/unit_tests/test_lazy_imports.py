"""Tests that heavy optional dependencies are not imported eagerly.

Regression test for https://github.com/langchain-ai/langchain/issues/35437
Importing ``langchain_text_splitters`` used to pull in spacy / nltk /
sentence-transformers (and transitively torch) at module level, adding ~700 MiB
of RSS even when only lightweight splitters were needed.
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Unit: heavy modules stay out of sys.modules on bare package import
# ---------------------------------------------------------------------------

_HEAVY_MODULES = ("nltk", "spacy", "torch", "sentence_transformers")


def _fresh_import_text_splitters() -> None:
    """Force-reimport langchain_text_splitters from scratch."""
    # Remove cached package + submodules so the import runs top-level code again.
    to_remove = [
        key
        for key in sys.modules
        if key == "langchain_text_splitters" or key.startswith("langchain_text_splitters.")
    ]
    for key in to_remove:
        del sys.modules[key]
    importlib.import_module("langchain_text_splitters")


def test_heavy_deps_not_imported_eagerly() -> None:
    """Importing the package must NOT pull in nltk / spacy / torch / sentence_transformers."""
    # Snapshot which heavy modules are already loaded (e.g. by the test runner).
    already_loaded = {m for m in _HEAVY_MODULES if m in sys.modules}

    _fresh_import_text_splitters()

    newly_loaded = {
        m for m in _HEAVY_MODULES if m in sys.modules and m not in already_loaded
    }
    assert newly_loaded == set(), (
        f"Heavy modules {newly_loaded} were imported eagerly by langchain_text_splitters. "
        "They should be lazily imported only when the corresponding splitter class is accessed."
    )


# ---------------------------------------------------------------------------
# Unit: lazy __getattr__ resolves the expected classes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "class_name",
    ["NLTKTextSplitter", "SpacyTextSplitter", "SentenceTransformersTokenTextSplitter"],
)
def test_lazy_getattr_resolves(class_name: str) -> None:
    """Accessing a lazily-loaded class via __getattr__ should return a class."""
    import langchain_text_splitters

    # Only test resolution when the underlying optional dep is installed.
    try:
        cls = getattr(langchain_text_splitters, class_name)
    except ImportError:
        pytest.skip(f"Optional dependency for {class_name} not installed")
    assert isinstance(cls, type), f"{class_name} should be a class, got {type(cls)}"


def test_getattr_raises_for_unknown() -> None:
    """Accessing a non-existent attribute should raise AttributeError."""
    import langchain_text_splitters

    with pytest.raises(AttributeError, match="no_such_thing"):
        _ = langchain_text_splitters.no_such_thing  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Eagerly-imported splitters still work
# ---------------------------------------------------------------------------

def test_eager_imports_still_accessible() -> None:
    """Lightweight splitters should remain directly importable."""
    from langchain_text_splitters import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
        TextSplitter,
    )

    assert issubclass(RecursiveCharacterTextSplitter, TextSplitter)
    assert issubclass(CharacterTextSplitter, TextSplitter)


# ---------------------------------------------------------------------------
# E2E: import package → use RecursiveCharacterTextSplitter without bloat
# ---------------------------------------------------------------------------

def test_e2e_lightweight_splitter_no_heavy_deps() -> None:
    """End-to-end: using RecursiveCharacterTextSplitter must not import heavy deps."""
    already_loaded = {m for m in _HEAVY_MODULES if m in sys.modules}

    _fresh_import_text_splitters()

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    result = splitter.split_text("Hello world. " * 50)
    assert len(result) >= 1

    newly_loaded = {
        m for m in _HEAVY_MODULES if m in sys.modules and m not in already_loaded
    }
    assert newly_loaded == set(), (
        f"Using RecursiveCharacterTextSplitter pulled in heavy modules: {newly_loaded}"
    )
