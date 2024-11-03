"""**Example selector** implements logic for selecting examples to include them
in prompts.
This allows us to select examples that are most relevant to the input.

There could be multiple strategies for selecting examples. For example, one could
select examples based on the similarity of the input to the examples. Another
strategy could be to select examples based on the diversity of the examples.
"""

from langchain_community.example_selectors.ngram_overlap import (
    NGramOverlapExampleSelector,
    ngram_overlap_score,
)

__all__ = [
    "NGramOverlapExampleSelector",
    "ngram_overlap_score",
]
