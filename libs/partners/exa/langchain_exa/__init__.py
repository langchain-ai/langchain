"""LangChain integration for Exa."""

from exa_py.api import (
    HighlightsContentsOptions,
    TextContentsOptions,
)

from langchain_exa._version import __version__
from langchain_exa.retrievers import ExaSearchRetriever
from langchain_exa.tools import ExaFindSimilarResults, ExaSearchResults

__all__ = [
    "ExaFindSimilarResults",
    "ExaSearchResults",
    "ExaSearchRetriever",
    "HighlightsContentsOptions",
    "TextContentsOptions",
    "__version__",
]
