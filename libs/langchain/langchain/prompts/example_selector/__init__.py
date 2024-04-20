"""Logic for selecting examples to include in prompts."""
from langchain_community.example_selectors.ngram_overlap import (
    NGramOverlapExampleSelector,
)
from langchain_core.example_selectors.length_based import (
    LengthBasedExampleSelector,
)
from langchain_core.example_selectors.semantic_similarity import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)

__all__ = [
    "LengthBasedExampleSelector",
    "MaxMarginalRelevanceExampleSelector",
    "NGramOverlapExampleSelector",
    "SemanticSimilarityExampleSelector",
]
