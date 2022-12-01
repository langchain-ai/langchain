"""Example selector that selects examples based on SemanticSimilarity."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra

from langchain.embeddings.base import Embeddings
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.vectorstores.base import VectorStore


def sorted_values(values: Dict[str, str]) -> List[Any]:
    """Return a list of values in dict sorted by key."""
    return [values[val] for val in sorted(values)]


class SemanticSimilarityExampleSelector(BaseExampleSelector, BaseModel):
    """Example selector that selects examples based on SemanticSimilarity."""

    vectorstore: VectorStore
    """VectorStore than contains information about examples."""
    k: int = 4
    """Number of examples to select."""
    example_keys: Optional[List[str]] = None
    """Optional keys to filter examples to."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to vectorstore."""
        string_example = " ".join(sorted_values(example))
        self.vectorstore.add_texts([string_example], metadatas=[example])

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on semantic similarity."""
        # Get the docs with the highest similarity.
        query = " ".join(sorted_values(input_variables))
        example_docs = self.vectorstore.similarity_search(query, k=self.k)
        # Get the examples from the metadata.
        # This assumes that examples are stored in metadata.
        examples = [dict(e.metadata) for e in example_docs]
        # If example keys are provided, filter examples to those keys.
        if self.example_keys:
            examples = [{k: eg[k] for k in self.example_keys} for eg in examples]
        return examples

    @classmethod
    def from_examples(
        cls,
        examples: List[dict],
        embeddings: Embeddings,
        vectorstore_cls: VectorStore,
        k: int = 4,
        **vectorstore_cls_kwargs: Any,
    ) -> SemanticSimilarityExampleSelector:
        """Create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on query similarity.

        Args:
            examples: List of examples to use in the prompt.
            suffix: String to go after the list of examples. Should generally
                set up the user's input.
            input_variables: A list of variable names the final prompt template
                will expect.
            embeddings: An iniialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            example_separator: The seperator to use in between examples. Defaults
                to two new line characters.
            prefix: String that should go before any examples. Generally includes
                examples. Default to an empty string.
            k: Number of examples to select
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        """
        string_examples = [" ".join(sorted_values(eg)) for eg in examples]
        vectorstore = vectorstore_cls.from_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(vectorstore=vectorstore, k=k)
