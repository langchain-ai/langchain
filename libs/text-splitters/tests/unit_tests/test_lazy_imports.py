"""Tests that heavy optional dependencies are not imported eagerly.

Regression test for https://github.com/langchain-ai/langchain/issues/35437.
Importing ``langchain_text_splitters`` used to pull in spacy / nltk /
sentence-transformers (and transitively torch) at module level, adding ~700 MiB
of RSS even when only lightweight splitters were needed.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_HEAVY_MODULES = ("spacy", "sentence_transformers")


def _fresh_import_text_splitters() -> None:
    """Force-reimport langchain_text_splitters from scratch."""
    to_remove = [
        key
        for key in sys.modules
        if key == "langchain_text_splitters"
        or key.startswith("langchain_text_splitters.")
    ]
    for key in to_remove:
        del sys.modules[key]
    importlib.import_module("langchain_text_splitters")


def _module() -> ModuleType:
    return importlib.import_module("langchain_text_splitters")


def test_heavy_deps_not_imported_eagerly() -> None:
    """Importing package must not eagerly import heavy optional deps."""
    already_loaded = {m for m in _HEAVY_MODULES if m in sys.modules}

    _fresh_import_text_splitters()

    newly_loaded = {
        m for m in _HEAVY_MODULES if m in sys.modules and m not in already_loaded
    }
    assert newly_loaded == set(), (
        "Heavy modules were imported eagerly by langchain_text_splitters: "
        f"{newly_loaded}. They should be imported lazily only when needed."
    )


@pytest.mark.parametrize(
    "class_name",
    ["NLTKTextSplitter", "SpacyTextSplitter", "SentenceTransformersTokenTextSplitter"],
)
def test_lazy_getattr_resolves(class_name: str) -> None:
    mod = _module()

    try:
        cls = getattr(mod, class_name)
    except ImportError:
        pytest.skip(f"Optional dependency for {class_name} not installed")
    assert isinstance(cls, type), f"{class_name} should be a class, got {type(cls)}"


def test_getattr_raises_for_unknown() -> None:
    mod = _module()

    with pytest.raises(AttributeError, match="no_such_thing"):
        _ = mod.no_such_thing


def test_eager_imports_still_accessible() -> None:
    _fresh_import_text_splitters()
    mod = _module()

    assert issubclass(mod.RecursiveCharacterTextSplitter, mod.TextSplitter)
    assert issubclass(mod.CharacterTextSplitter, mod.TextSplitter)


def test_e2e_lightweight_splitter_no_heavy_deps() -> None:
    already_loaded = {m for m in _HEAVY_MODULES if m in sys.modules}

    _fresh_import_text_splitters()
    mod = _module()

    splitter = mod.RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    result = splitter.split_text("Hello world. " * 50)
    assert len(result) >= 1

    newly_loaded = {
        m for m in _HEAVY_MODULES if m in sys.modules and m not in already_loaded
    }
    assert newly_loaded == set(), (
        f"Using RecursiveCharacterTextSplitter imported heavy modules: {newly_loaded}"
    )
