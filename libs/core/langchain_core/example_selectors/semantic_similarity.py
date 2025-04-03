"""Example selector that selects examples based on SemanticSimilarity."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict

from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings


def sorted_values(values: dict[str, str]) -> list[Any]:
    """Return a list of values in dict sorted by key.

    Args:
        values: A dictionary with keys as input variables
            and values as their values.

    Returns:
        A list of values in dict sorted by key.
    """
    return [values[val] for val in sorted(values)]


class _VectorStoreExampleSelector(BaseExampleSelector, BaseModel, ABC):
    """Example selector that selects examples based on SemanticSimilarity."""

    vectorstore: VectorStore
    """VectorStore that contains information about examples."""
    k: int = 4
    """Number of examples to select."""
    example_keys: Optional[list[str]] = None
    """Optional keys to filter examples to."""
    input_keys: Optional[list[str]] = None
    """Optional keys to filter input to. If provided, the search is based on
    the input variables instead of all variables."""
    vectorstore_kwargs: Optional[dict[str, Any]] = None
    """Extra arguments passed to similarity_search function of the vectorstore."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @staticmethod
    def _example_to_text(
        example: dict[str, str], input_keys: Optional[list[str]]
    ) -> str:
        if input_keys:
            return " ".join(sorted_values({key: example[key] for key in input_keys}))
        return " ".join(sorted_values(example))

    def _documents_to_examples(self, documents: list[Document]) -> list[dict]:
        # Get the examples from the metadata.
        # This assumes that examples are stored in metadata.
        examples = [dict(e.metadata) for e in documents]
        # If example keys are provided, filter examples to those keys.
        if self.example_keys:
            examples = [{k: eg[k] for k in self.example_keys} for eg in examples]
        return examples

    def add_example(self, example: dict[str, str]) -> str:
        """Add a new example to vectorstore.

        Args:
            example: A dictionary with keys as input variables
                and values as their values.

        Returns:
            The ID of the added example.
        """
        ids = self.vectorstore.add_texts(
            [self._example_to_text(example, self.input_keys)], metadatas=[example]
        )
        return ids[0]

    async def aadd_example(self, example: dict[str, str]) -> str:
        """Async add new example to vectorstore.

        Args:
            example: A dictionary with keys as input variables
                and values as their values.

        Returns:
            The ID of the added example.
        """
        ids = await self.vectorstore.aadd_texts(
            [self._example_to_text(example, self.input_keys)], metadatas=[example]
        )
        return ids[0]


class SemanticSimilarityExampleSelector(_VectorStoreExampleSelector):
    """Select examples based on semantic similarity."""

    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Select examples based on semantic similarity.

        Args:
            input_variables: The input variables to use for search.

        Returns:
            The selected examples.
        """
        # Get the docs with the highest similarity.
        vectorstore_kwargs = self.vectorstore_kwargs or {}
        example_docs = self.vectorstore.similarity_search(
            self._example_to_text(input_variables, self.input_keys),
            k=self.k,
            **vectorstore_kwargs,
        )
        return self._documents_to_examples(example_docs)

    async def aselect_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Asynchronously select examples based on semantic similarity.

        Args:
            input_variables: The input variables to use for search.

        Returns:
            The selected examples.
        """
        # Get the docs with the highest similarity.
        vectorstore_kwargs = self.vectorstore_kwargs or {}
        example_docs = await self.vectorstore.asimilarity_search(
            self._example_to_text(input_variables, self.input_keys),
            k=self.k,
            **vectorstore_kwargs,
        )
        return self._documents_to_examples(example_docs)

    @classmethod
    def from_examples(
        cls,
        examples: list[dict],
        embeddings: Embeddings,
        vectorstore_cls: type[VectorStore],
        k: int = 4,
        input_keys: Optional[list[str]] = None,
        *,
        example_keys: Optional[list[str]] = None,
        vectorstore_kwargs: Optional[dict] = None,
        **vectorstore_cls_kwargs: Any,
    ) -> SemanticSimilarityExampleSelector:
        """Create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on query similarity.

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select. Default is 4.
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            example_keys: If provided, keys to filter examples to.
            vectorstore_kwargs: Extra arguments passed to similarity_search function
                of the vectorstore.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        """
        string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]
        vectorstore = vectorstore_cls.from_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(
            vectorstore=vectorstore,
            k=k,
            input_keys=input_keys,
            example_keys=example_keys,
            vectorstore_kwargs=vectorstore_kwargs,
        )

    @classmethod
    async def afrom_examples(
        cls,
        examples: list[dict],
        embeddings: Embeddings,
        vectorstore_cls: type[VectorStore],
        k: int = 4,
        input_keys: Optional[list[str]] = None,
        *,
        example_keys: Optional[list[str]] = None,
        vectorstore_kwargs: Optional[dict] = None,
        **vectorstore_cls_kwargs: Any,
    ) -> SemanticSimilarityExampleSelector:
        """Async create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on query similarity.

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select. Default is 4.
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            example_keys: If provided, keys to filter examples to.
            vectorstore_kwargs: Extra arguments passed to similarity_search function
                of the vectorstore.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        """
        string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]
        vectorstore = await vectorstore_cls.afrom_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(
            vectorstore=vectorstore,
            k=k,
            input_keys=input_keys,
            example_keys=example_keys,
            vectorstore_kwargs=vectorstore_kwargs,
        )


class MaxMarginalRelevanceExampleSelector(_VectorStoreExampleSelector):
    """Select examples based on Max Marginal Relevance.

    This was shown to improve performance in this paper:
    https://arxiv.org/pdf/2211.13892.pdf
    """

    fetch_k: int = 20
    """Number of examples to fetch to rerank."""

    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Select examples based on Max Marginal Relevance.

        Args:
            input_variables: The input variables to use for search.

        Returns:
            The selected examples.
        """
        example_docs = self.vectorstore.max_marginal_relevance_search(
            self._example_to_text(input_variables, self.input_keys),
            k=self.k,
            fetch_k=self.fetch_k,
        )
        return self._documents_to_examples(example_docs)

    async def aselect_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Asynchronously select examples based on Max Marginal Relevance.

        Args:
            input_variables: The input variables to use for search.

        Returns:
            The selected examples.
        """
        example_docs = await self.vectorstore.amax_marginal_relevance_search(
            self._example_to_text(input_variables, self.input_keys),
            k=self.k,
            fetch_k=self.fetch_k,
        )
        return self._documents_to_examples(example_docs)

    @classmethod
    def from_examples(
        cls,
        examples: list[dict],
        embeddings: Embeddings,
        vectorstore_cls: type[VectorStore],
        k: int = 4,
        input_keys: Optional[list[str]] = None,
        fetch_k: int = 20,
        example_keys: Optional[list[str]] = None,
        vectorstore_kwargs: Optional[dict] = None,
        **vectorstore_cls_kwargs: Any,
    ) -> MaxMarginalRelevanceExampleSelector:
        """Create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on Max Marginal Relevance.

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select. Default is 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                Default is 20.
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            example_keys: If provided, keys to filter examples to.
            vectorstore_kwargs: Extra arguments passed to similarity_search function
                of the vectorstore.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        """
        string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]
        vectorstore = vectorstore_cls.from_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(
            vectorstore=vectorstore,
            k=k,
            fetch_k=fetch_k,
            input_keys=input_keys,
            example_keys=example_keys,
            vectorstore_kwargs=vectorstore_kwargs,
        )

    @classmethod
    async def afrom_examples(
        cls,
        examples: list[dict],
        embeddings: Embeddings,
        vectorstore_cls: type[VectorStore],
        *,
        k: int = 4,
        input_keys: Optional[list[str]] = None,
        fetch_k: int = 20,
        example_keys: Optional[list[str]] = None,
        vectorstore_kwargs: Optional[dict] = None,
        **vectorstore_cls_kwargs: Any,
    ) -> MaxMarginalRelevanceExampleSelector:
        """Create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on Max Marginal Relevance.

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select. Default is 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                Default is 20.
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            example_keys: If provided, keys to filter examples to.
            vectorstore_kwargs: Extra arguments passed to similarity_search function
                of the vectorstore.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        """
        string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]
        vectorstore = await vectorstore_cls.afrom_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(
            vectorstore=vectorstore,
            k=k,
            fetch_k=fetch_k,
            input_keys=input_keys,
            example_keys=example_keys,
            vectorstore_kwargs=vectorstore_kwargs,
        )
