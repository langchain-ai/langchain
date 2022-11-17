"""Example selector that selects examples based on SemanticSimilarity."""
from typing import Dict, List, Optional

from pydantic import BaseModel

from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.vectorstores.base import VectorStore


class SemanticSimilarityExampleSelector(BaseExampleSelector, BaseModel):
    """Example selector that selects examples based on SemanticSimilarity."""

    vectorstore: VectorStore
    """VectorStore than contains information about examples."""
    k: int = 4
    """Number of examples to select."""
    example_keys: Optional[List[str]] = None
    """Optional keys to filter examples to."""

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on semantic similarity."""
        # Get the docs with the highest similarity.
        query = " ".join([v for k, v in input_variables.items()])
        example_docs = self.vectorstore.similarity_search(query, k=self.k)
        # Get the examples from the metadata.
        # This assumes that examples are stored in metadata.
        examples = [dict(e.metadata) for e in example_docs]
        # If example keys are provided, filter examples to those keys.
        if self.example_keys:
            examples = [{k: eg[k] for k in self.example_keys} for eg in examples]
        return examples
