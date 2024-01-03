"""Example selector that selects examples based on SemanticSimilarity."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.pydantic_v1 import BaseModel, Extra
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


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
    input_keys: Optional[List[str]] = None
    """Optional keys to filter input to. If provided, the search is based on
    the input variables instead of all variables."""
    vectorstore_kwargs: Optional[Dict[str, Any]] = None
    """Extra arguments passed to similarity_search function of the vectorstore."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def add_example(self, example: Dict[str, str]) -> str:
        """Add new example to vectorstore."""
        if self.input_keys:
            string_example = " ".join(
                sorted_values({key: example[key] for key in self.input_keys})
            )
        else:
            string_example = " ".join(sorted_values(example))
        ids = self.vectorstore.add_texts([string_example], metadatas=[example])
        return ids[0]

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on semantic similarity."""
        # Get the docs with the highest similarity.
        if self.input_keys:
            input_variables = {key: input_variables[key] for key in self.input_keys}
        vectorstore_kwargs = self.vectorstore_kwargs or {}
        query = " ".join(sorted_values(input_variables))
        example_docs = self.vectorstore.similarity_search(
            query, k=self.k, **vectorstore_kwargs
        )
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
        vectorstore_cls: Type[VectorStore],
        k: int = 4,
        input_keys: Optional[List[str]] = None,
        **vectorstore_cls_kwargs: Any,
    ) -> SemanticSimilarityExampleSelector:
        """Create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on query similarity.

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        """
        if input_keys:
            string_examples = [
                " ".join(sorted_values({k: eg[k] for k in input_keys}))
                for eg in examples
            ]
        else:
            string_examples = [" ".join(sorted_values(eg)) for eg in examples]
        vectorstore = vectorstore_cls.from_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(vectorstore=vectorstore, k=k, input_keys=input_keys)


class MaxMarginalRelevanceExampleSelector(SemanticSimilarityExampleSelector):
    """ExampleSelector that selects examples based on Max Marginal Relevance.

    This was shown to improve performance in this paper:
    https://arxiv.org/pdf/2211.13892.pdf
    """

    fetch_k: int = 20
    """Number of examples to fetch to rerank."""

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on semantic similarity."""
        # Get the docs with the highest similarity.
        if self.input_keys:
            input_variables = {key: input_variables[key] for key in self.input_keys}
        query = " ".join(sorted_values(input_variables))
        example_docs = self.vectorstore.max_marginal_relevance_search(
            query, k=self.k, fetch_k=self.fetch_k
        )
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
        vectorstore_cls: Type[VectorStore],
        k: int = 4,
        input_keys: Optional[List[str]] = None,
        fetch_k: int = 20,
        **vectorstore_cls_kwargs: Any,
    ) -> MaxMarginalRelevanceExampleSelector:
        """Create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on query similarity.

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An iniialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        """
        if input_keys:
            string_examples = [
                " ".join(sorted_values({k: eg[k] for k in input_keys}))
                for eg in examples
            ]
        else:
            string_examples = [" ".join(sorted_values(eg)) for eg in examples]
        vectorstore = vectorstore_cls.from_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(vectorstore=vectorstore, k=k, fetch_k=fetch_k, input_keys=input_keys)
