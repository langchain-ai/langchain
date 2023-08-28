"""Logic for selecting examples to include in prompts."""
from langchain_xfyun.prompts.example_selector.length_based import LengthBasedExampleSelector
from langchain_xfyun.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector
from langchain_xfyun.prompts.example_selector.semantic_similarity import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)

__all__ = [
    "LengthBasedExampleSelector",
    "MaxMarginalRelevanceExampleSelector",
    "NGramOverlapExampleSelector",
    "SemanticSimilarityExampleSelector",
]
