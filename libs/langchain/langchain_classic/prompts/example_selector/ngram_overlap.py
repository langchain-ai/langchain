from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.example_selectors.ngram_overlap import (
        NGramOverlapExampleSelector,
        ngram_overlap_score,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
MODULE_LOOKUP = {
    "NGramOverlapExampleSelector": (
        "langchain_community.example_selectors.ngram_overlap"
    ),
    "ngram_overlap_score": "langchain_community.example_selectors.ngram_overlap",
}

_import_attribute = create_importer(__file__, deprecated_lookups=MODULE_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "NGramOverlapExampleSelector",
    "ngram_overlap_score",
]
