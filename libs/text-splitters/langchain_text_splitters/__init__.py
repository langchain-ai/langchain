"""Text Splitters are classes for splitting text.

!!! note

    `MarkdownHeaderTextSplitter` and `HTMLHeaderTextSplitter` do not derive from
    `TextSplitter`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_text_splitters.base import (
    Language,
    TextSplitter,
    Tokenizer,
    TokenTextSplitter,
    split_text_on_tokens,
)
from langchain_text_splitters.character import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_text_splitters.html import (
    ElementType,
    HTMLHeaderTextSplitter,
    HTMLSectionSplitter,
    HTMLSemanticPreservingSplitter,
)
from langchain_text_splitters.json import RecursiveJsonSplitter
from langchain_text_splitters.jsx import JSFrameworkTextSplitter
from langchain_text_splitters.konlpy import KonlpyTextSplitter
from langchain_text_splitters.latex import LatexTextSplitter
from langchain_text_splitters.markdown import (
    ExperimentalMarkdownSyntaxTextSplitter,
    HeaderType,
    LineType,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
)
from langchain_text_splitters.python import PythonCodeTextSplitter

if TYPE_CHECKING:
    from langchain_text_splitters.nltk import NLTKTextSplitter
    from langchain_text_splitters.sentence_transformers import (
        SentenceTransformersTokenTextSplitter,
    )
    from langchain_text_splitters.spacy import SpacyTextSplitter

__all__ = [
    "CharacterTextSplitter",
    "ElementType",
    "ExperimentalMarkdownSyntaxTextSplitter",
    "HTMLHeaderTextSplitter",
    "HTMLSectionSplitter",
    "HTMLSemanticPreservingSplitter",
    "HeaderType",
    "JSFrameworkTextSplitter",
    "KonlpyTextSplitter",
    "Language",
    "LatexTextSplitter",
    "LineType",
    "MarkdownHeaderTextSplitter",
    "MarkdownTextSplitter",
    "NLTKTextSplitter",
    "PythonCodeTextSplitter",
    "RecursiveCharacterTextSplitter",
    "RecursiveJsonSplitter",
    "SentenceTransformersTokenTextSplitter",
    "SpacyTextSplitter",
    "TextSplitter",
    "TokenTextSplitter",
    "Tokenizer",
    "split_text_on_tokens",
]

# Lazy imports for modules with heavy transitive dependencies (torch, spacy, nltk).
# Importing these eagerly at package level pulls in ~700MB of memory even when users
# only need lightweight splitters like RecursiveCharacterTextSplitter.
# See: https://github.com/langchain-ai/langchain/issues/35437
_LAZY_IMPORTS: dict[str, str] = {
    "NLTKTextSplitter": "nltk",
    "SpacyTextSplitter": "spacy",
    "SentenceTransformersTokenTextSplitter": "sentence_transformers",
}


def __getattr__(attr_name: str) -> object:
    module_name = _LAZY_IMPORTS.get(attr_name)
    if module_name is not None:
        from importlib import import_module

        module = import_module(f".{module_name}", __spec__.parent)
        result = getattr(module, attr_name)
        globals()[attr_name] = result
        return result
    msg = f"module {__name__!r} has no attribute {attr_name!r}"
    raise AttributeError(msg)
