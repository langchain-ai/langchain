"""Logic for selecting examples to include in prompts."""

from typing import TYPE_CHECKING, Any

from langchain_core.example_selectors.length_based import (
    LengthBasedExampleSelector,
)
from langchain_core.example_selectors.semantic_similarity import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.example_selectors.ngram_overlap import (
        NGramOverlapExampleSelector,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUPS = {
    "NGramOverlapExampleSelector": (
        "langchain_community.example_selectors.ngram_overlap"
    ),
}

_import_attribute = create_importer(__file__, deprecated_lookups=DEPRECATED_LOOKUPS)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "LengthBasedExampleSelector",
    "MaxMarginalRelevanceExampleSelector",
    "NGramOverlapExampleSelector",
    "SemanticSimilarityExampleSelector",
]
