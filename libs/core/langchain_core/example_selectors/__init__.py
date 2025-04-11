"""Example selectors.

**Example selector** implements logic for selecting examples to include them in prompts.
This allows us to select examples that are most relevant to the input.
"""

from typing import TYPE_CHECKING

from langchain_core._lazy_imports import create_dynamic_getattr

if TYPE_CHECKING:
    from langchain_core.example_selectors.base import BaseExampleSelector
    from langchain_core.example_selectors.length_based import (
        LengthBasedExampleSelector,
    )
    from langchain_core.example_selectors.semantic_similarity import (
        MaxMarginalRelevanceExampleSelector,
        SemanticSimilarityExampleSelector,
        sorted_values,
    )

__all__ = [
    "BaseExampleSelector",
    "LengthBasedExampleSelector",
    "MaxMarginalRelevanceExampleSelector",
    "SemanticSimilarityExampleSelector",
    "sorted_values",
]

__getattr__ = create_dynamic_getattr(
    package_name="langchain_core",
    module_path="example_selectors",
    dynamic_imports={
        "BaseExampleSelector": "base",
        "LengthBasedExampleSelector": "length_based",
        "MaxMarginalRelevanceExampleSelector": "semantic_similarity",
        "SemanticSimilarityExampleSelector": "semantic_similarity",
        "sorted_values": "semantic_similarity",
    },
)


def __dir__() -> list[str]:
    return list(__all__)
