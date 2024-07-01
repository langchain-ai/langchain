"""**Example selector** implements logic for selecting examples to include them
in prompts.
This allows us to select examples that are most relevant to the input.
"""

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
