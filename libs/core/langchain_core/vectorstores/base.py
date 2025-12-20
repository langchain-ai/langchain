"""**Vector store** stores embedded data and performs vector search.

One of the most common ways to store and search over unstructured data is to
embed it and store the resulting embedding vectors, and then query the store
and retrieve the data that are 'most similar' to the embedded query.
"""

from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from itertools import cycle
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    TypeVar,
)

from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self, override

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever, LangSmithRetrieverParams
from langchain_core.runnables.config import run_in_executor

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator, Sequence

    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )

logger = logging.getLogger(__name__)

VST = TypeVar("VST", bound="VectorStore")


class VectorStore(ABC):
    """Interface for vector store."""

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the `VectorStore`.

        Args:
            texts: Iterable of strings to add to the `VectorStore`.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
            **kwargs: `VectorStore` specific parameters.
                One of the kwargs should be `ids` which is a list of ids
                associated with the texts.

        Returns:
            List of IDs from adding the texts into the `VectorStore`.

        Raises:
            ValueError: If the number of metadatas does not match the number of texts.
            ValueError: If the number of IDs does not match the number of texts.
        """
        if type(self).add_documents != VectorStore.add_documents:
            # This condition is triggered if the subclass has provided
            # an implementation of the upsert method.
            # The existing add_texts
            texts_: Sequence[str] = (
                texts if isinstance(texts, (list, tuple)) else list(texts)
            )
            if metadatas and len(metadatas) != len(texts_):
                msg = (
                    "The number of metadatas must match the number of texts."
                    f"Got {len(metadatas)} metadatas and {len(texts_)} texts."
                )
                raise ValueError(msg)
            metadatas_ = iter(metadatas) if metadatas else cycle([{}])
            ids_: Iterator[str | None] = iter(ids) if ids else cycle([None])
            docs = [
                Document(id=id_, page_content=text, metadata=metadata_)
                for text, metadata_, id_ in zip(texts, metadatas_, ids_, strict=False)
            ]
            if ids is not None:
                # For backward compatibility
                kwargs["ids"] = ids

            return self.add_documents(docs, **kwargs)
        msg = f"`add_texts` has not been implemented for {self.__class__.__name__} "
        raise NotImplementedError(msg)

    @property
    def embeddings(self) -> Embeddings | None:
        """Access the query embedding object if available."""
        logger.debug(
            "The embeddings property has not been implemented for %s",
            self.__class__.__name__,
        )
        return None

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of IDs to delete. If `None`, delete all.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            `True` if deletion is successful, `False` otherwise, `None` if not
                implemented.
        """
        msg = "delete method must be implemented by subclass."
        raise NotImplementedError(msg)

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their IDs.

        The returned documents are expected to have the ID field set to the ID of the
        document in the vector store.

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of IDs to retrieve.

        Returns:
            List of `Document` objects.
        """
        msg = f"{self.__class__.__name__} does not yet support get_by_ids."
        raise NotImplementedError(msg)

    # Implementations should override this method to provide an async native version.
    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Async get documents by their IDs.

        The returned documents are expected to have the ID field set to the ID of the
        document in the vector store.

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of IDs to retrieve.

        Returns:
            List of `Document` objects.
        """
        return await run_in_executor(None, self.get_by_ids, ids)

    async def adelete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Async delete by vector ID or other criteria.

        Args:
            ids: List of IDs to delete. If `None`, delete all.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            `True` if deletion is successful, `False` otherwise, `None` if not
                implemented.
        """
        return await run_in_executor(None, self.delete, ids, **kwargs)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Async run more texts through the embeddings and add to the `VectorStore`.

        Args:
            texts: Iterable of strings to add to the `VectorStore`.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list
            **kwargs: `VectorStore` specific parameters.

        Returns:
            List of IDs from adding the texts into the `VectorStore`.

        Raises:
            ValueError: If the number of metadatas does not match the number of texts.
            ValueError: If the number of IDs does not match the number of texts.
        """
        if ids is not None:
            # For backward compatibility
            kwargs["ids"] = ids
        if type(self).aadd_documents != VectorStore.aadd_documents:
            # This condition is triggered if the subclass has provided
            # an implementation of the upsert method.
            # The existing add_texts
            texts_: Sequence[str] = (
                texts if isinstance(texts, (list, tuple)) else list(texts)
            )
            if metadatas and len(metadatas) != len(texts_):
                msg = (
                    "The number of metadatas must match the number of texts."
                    f"Got {len(metadatas)} metadatas and {len(texts_)} texts."
                )
                raise ValueError(msg)
            metadatas_ = iter(metadatas) if metadatas else cycle([{}])
            ids_: Iterator[str | None] = iter(ids) if ids else cycle([None])

            docs = [
                Document(id=id_, page_content=text, metadata=metadata_)
                for text, metadata_, id_ in zip(texts, metadatas_, ids_, strict=False)
            ]
            return await self.aadd_documents(docs, **kwargs)
        return await run_in_executor(None, self.add_texts, texts, metadatas, **kwargs)

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Add or update documents in the `VectorStore`.

        Args:
            documents: Documents to add to the `VectorStore`.
            **kwargs: Additional keyword arguments.

                If kwargs contains IDs and documents contain ids, the IDs in the kwargs
                will receive precedence.

        Returns:
            List of IDs of the added texts.
        """
        if type(self).add_texts != VectorStore.add_texts:
            if "ids" not in kwargs:
                ids = [doc.id for doc in documents]

                # If there's at least one valid ID, we'll assume that IDs
                # should be used.
                if any(ids):
                    kwargs["ids"] = ids

            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            return self.add_texts(texts, metadatas, **kwargs)
        msg = (
            f"`add_documents` and `add_texts` has not been implemented "
            f"for {self.__class__.__name__} "
        )
        raise NotImplementedError(msg)

    async def aadd_documents(
        self, documents: list[Document], **kwargs: Any
    ) -> list[str]:
        """Async run more documents through the embeddings and add to the `VectorStore`.

        Args:
            documents: Documents to add to the `VectorStore`.
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs of the added texts.
        """
        # If the async method has been overridden, we'll use that.
        if type(self).aadd_texts != VectorStore.aadd_texts:
            if "ids" not in kwargs:
                ids = [doc.id for doc in documents]

                # If there's at least one valid ID, we'll assume that IDs
                # should be used.
                if any(ids):
                    kwargs["ids"] = ids

            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            return await self.aadd_texts(texts, metadatas, **kwargs)

        return await run_in_executor(None, self.add_documents, documents, **kwargs)

    def search(self, query: str, search_type: str, **kwargs: Any) -> list[Document]:
        """Return docs most similar to query using a specified search type.

        Args:
            query: Input text.
            search_type: Type of search to perform.

                Can be `'similarity'`, `'mmr'`, or `'similarity_score_threshold'`.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of `Document` objects most similar to the query.

        Raises:
            ValueError: If `search_type` is not one of `'similarity'`,
                `'mmr'`, or `'similarity_score_threshold'`.
        """
        if search_type == "similarity":
            return self.similarity_search(query, **kwargs)
        if search_type == "similarity_score_threshold":
            docs_and_similarities = self.similarity_search_with_relevance_scores(
                query, **kwargs
            )
            return [doc for doc, _ in docs_and_similarities]
        if search_type == "mmr":
            return self.max_marginal_relevance_search(query, **kwargs)
        msg = (
            f"search_type of {search_type} not allowed. Expected "
            "search_type to be 'similarity', 'similarity_score_threshold'"
            " or 'mmr'."
        )
        raise ValueError(msg)

    async def asearch(
        self, query: str, search_type: str, **kwargs: Any
    ) -> list[Document]:
        """Async return docs most similar to query using a specified search type.

        Args:
            query: Input text.
            search_type: Type of search to perform.

                Can be `'similarity'`, `'mmr'`, or `'similarity_score_threshold'`.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of `Document` objects most similar to the query.

        Raises:
            ValueError: If `search_type` is not one of `'similarity'`,
                `'mmr'`, or `'similarity_score_threshold'`.
        """
        if search_type == "similarity":
            return await self.asimilarity_search(query, **kwargs)
        if search_type == "similarity_score_threshold":
            docs_and_similarities = await self.asimilarity_search_with_relevance_scores(
                query, **kwargs
            )
            return [doc for doc, _ in docs_and_similarities]
        if search_type == "mmr":
            return await self.amax_marginal_relevance_search(query, **kwargs)
        msg = (
            f"search_type of {search_type} not allowed. Expected "
            "search_type to be 'similarity', 'similarity_score_threshold' or 'mmr'."
        )
        raise ValueError(msg)

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Input text.
            k: Number of `Document` objects to return.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of `Document` objects most similar to the query.
        """

    @staticmethod
    def _euclidean_relevance_score_fn(distance: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        # The 'correct' relevance function
        # may differ depending on a few things, including:
        # - the distance / similarity metric used by the VectorStore
        # - the scale of your embeddings (OpenAI's are unit normed. Many
        #  others are not!)
        # - embedding dimensionality
        # - etc.
        # This function converts the Euclidean norm of normalized embeddings
        # (0 is most similar, sqrt(2) most dissimilar)
        # to a similarity function (0 to 1)
        return 1.0 - distance / math.sqrt(2)

    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""
        return 1.0 - distance

    @staticmethod
    def _max_inner_product_relevance_score_fn(distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""
        if distance > 0:
            return 1.0 - distance

        return -1.0 * distance

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """The 'correct' relevance function.

        may differ depending on a few things, including:

        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.

        Vectorstores should define their own selection-based method of relevance.
        """
        raise NotImplementedError

    def similarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Run similarity search with distance.

        Args:
            *args: Arguments to pass to the search method.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of tuples of `(doc, similarity_score)`.
        """
        raise NotImplementedError

    async def asimilarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Async run similarity search with distance.

        Args:
            *args: Arguments to pass to the search method.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of tuples of `(doc, similarity_score)`.
        """
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        return await run_in_executor(
            None, self.similarity_search_with_score, *args, **kwargs
        )

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Default similarity search with relevance scores.

        Modify if necessary in subclass.
        Return docs and relevance scores in the range `[0, 1]`.

        `0` is dissimilar, `1` is most similar.

        Args:
            query: Input text.
            k: Number of `Document` objects to return.
            **kwargs: Kwargs to be passed to similarity search.

                Should include `score_threshold`, an optional floating point value
                between `0` to `1` to filter the resulting set of retrieved docs.

        Returns:
            List of tuples of `(doc, similarity_score)`
        """
        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]

    async def _asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Default similarity search with relevance scores.

        Modify if necessary in subclass.
        Return docs and relevance scores in the range `[0, 1]`.

        `0` is dissimilar, `1` is most similar.

        Args:
            query: Input text.
            k: Number of `Document` objects to return.
            **kwargs: Kwargs to be passed to similarity search.

                Should include `score_threshold`, an optional floating point value
                between `0` to `1` to filter the resulting set of retrieved docs.

        Returns:
            List of tuples of `(doc, similarity_score)`
        """
        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = await self.asimilarity_search_with_score(query, k, **kwargs)
        return [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and relevance scores in the range `[0, 1]`.

        `0` is dissimilar, `1` is most similar.

        Args:
            query: Input text.
            k: Number of `Document` objects to return.
            **kwargs: Kwargs to be passed to similarity search.

                Should include `score_threshold`, an optional floating point value
                between `0` to `1` to filter the resulting set of retrieved docs.

        Returns:
            List of tuples of `(doc, similarity_score)`.
        """
        score_threshold = kwargs.pop("score_threshold", None)

        docs_and_similarities = self._similarity_search_with_relevance_scores(
            query, k=k, **kwargs
        )
        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                "Relevance scores must be between"
                f" 0 and 1, got {docs_and_similarities}",
                stacklevel=2,
            )

        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                logger.warning(
                    "No relevant docs were retrieved using the "
                    "relevance score threshold %s",
                    score_threshold,
                )
        return docs_and_similarities

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Async return docs and relevance scores in the range `[0, 1]`.

        `0` is dissimilar, `1` is most similar.

        Args:
            query: Input text.
            k: Number of `Document` objects to return.
            **kwargs: Kwargs to be passed to similarity search.

                Should include `score_threshold`, an optional floating point value
                between `0` to `1` to filter the resulting set of retrieved docs.

        Returns:
            List of tuples of `(doc, similarity_score)`
        """
        score_threshold = kwargs.pop("score_threshold", None)

        docs_and_similarities = await self._asimilarity_search_with_relevance_scores(
            query, k=k, **kwargs
        )
        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                "Relevance scores must be between"
                f" 0 and 1, got {docs_and_similarities}",
                stacklevel=2,
            )

        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                logger.warning(
                    "No relevant docs were retrieved using the "
                    "relevance score threshold %s",
                    score_threshold,
                )
        return docs_and_similarities

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Async return docs most similar to query.

        Args:
            query: Input text.
            k: Number of `Document` objects to return.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of `Document` objects most similar to the query.
        """
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        return await run_in_executor(None, self.similarity_search, query, k=k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of `Document` objects to return.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of `Document` objects most similar to the query vector.
        """
        raise NotImplementedError

    async def asimilarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Async return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of `Document` objects to return.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of `Document` objects most similar to the query vector.
        """
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        return await run_in_executor(
            None, self.similarity_search_by_vector, embedding, k=k, **kwargs
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of `Document` objects to return.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree
                of diversity among the results with `0` corresponding
                to maximum diversity and `1` to minimum diversity.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of `Document` objects selected by maximal marginal relevance.
        """
        raise NotImplementedError

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Async return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of `Document` objects to return.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree
                of diversity among the results with `0` corresponding
                to maximum diversity and `1` to minimum diversity.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of `Document` objects selected by maximal marginal relevance.
        """
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search,
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of `Document` objects to return.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree
                of diversity among the results with `0` corresponding
                to maximum diversity and `1` to minimum diversity.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of `Document` objects selected by maximal marginal relevance.
        """
        raise NotImplementedError

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Async return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of `Document` objects to return.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree
                of diversity among the results with `0` corresponding
                to maximum diversity and `1` to minimum diversity.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of `Document` objects selected by maximal marginal relevance.
        """
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search_by_vector,
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> Self:
        """Return `VectorStore` initialized from documents and embeddings.

        Args:
            documents: List of `Document` objects to add to the `VectorStore`.
            embedding: Embedding function to use.
            **kwargs: Additional keyword arguments.

        Returns:
            `VectorStore` initialized from documents and embeddings.
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        if "ids" not in kwargs:
            ids = [doc.id for doc in documents]

            # If there's at least one valid ID, we'll assume that IDs
            # should be used.
            if any(ids):
                kwargs["ids"] = ids

        return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)

    @classmethod
    async def afrom_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> Self:
        """Async return `VectorStore` initialized from documents and embeddings.

        Args:
            documents: List of `Document` objects to add to the `VectorStore`.
            embedding: Embedding function to use.
            **kwargs: Additional keyword arguments.

        Returns:
            `VectorStore` initialized from documents and embeddings.
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        if "ids" not in kwargs:
            ids = [doc.id for doc in documents]

            # If there's at least one valid ID, we'll assume that IDs
            # should be used.
            if any(ids):
                kwargs["ids"] = ids

        return await cls.afrom_texts(texts, embedding, metadatas=metadatas, **kwargs)

    @classmethod
    @abstractmethod
    def from_texts(
        cls: type[VST],
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> VST:
        """Return `VectorStore` initialized from texts and embeddings.

        Args:
            texts: Texts to add to the `VectorStore`.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
            **kwargs: Additional keyword arguments.

        Returns:
            `VectorStore` initialized from texts and embeddings.
        """

    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Async return `VectorStore` initialized from texts and embeddings.

        Args:
            texts: Texts to add to the `VectorStore`.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
            **kwargs: Additional keyword arguments.

        Returns:
            `VectorStore` initialized from texts and embeddings.
        """
        if ids is not None:
            kwargs["ids"] = ids
        return await run_in_executor(
            None, cls.from_texts, texts, embedding, metadatas, **kwargs
        )

    def _get_retriever_tags(self) -> list[str]:
        """Get tags for retriever."""
        tags = [self.__class__.__name__]
        if self.embeddings:
            tags.append(self.embeddings.__class__.__name__)
        return tags

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        """Return `VectorStoreRetriever` initialized from this `VectorStore`.

        Args:
            **kwargs: Keyword arguments to pass to the search function.

                Can include:

                * `search_type`: Defines the type of search that the Retriever should
                    perform. Can be `'similarity'` (default), `'mmr'`, or
                    `'similarity_score_threshold'`.
                * `search_kwargs`: Keyword arguments to pass to the search function.

                    Can include things like:

                    * `k`: Amount of documents to return (Default: `4`)
                    * `score_threshold`: Minimum relevance threshold
                        for `similarity_score_threshold`
                    * `fetch_k`: Amount of documents to pass to MMR algorithm
                        (Default: `20`)
                    * `lambda_mult`: Diversity of results returned by MMR;
                        `1` for minimum diversity and 0 for maximum. (Default: `0.5`)
                    * `filter`: Filter by document metadata

        Returns:
            Retriever class for `VectorStore`.

        Examples:
        ```python
        # Retrieve more documents with higher diversity
        # Useful if your dataset has many similar documents
        docsearch.as_retriever(
            search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25}
        )

        # Fetch more documents for the MMR algorithm to consider
        # But only return the top 5
        docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 50})

        # Only retrieve documents that have a relevance score
        # Above a certain threshold
        docsearch.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.8},
        )

        # Only get the single most similar document from the dataset
        docsearch.as_retriever(search_kwargs={"k": 1})

        # Use a filter to only retrieve documents from a specific paper
        docsearch.as_retriever(
            search_kwargs={"filter": {"paper_title": "GPT-4 Technical Report"}}
        )
        ```
        """
        tags = kwargs.pop("tags", None) or [*self._get_retriever_tags()]
        return VectorStoreRetriever(vectorstore=self, tags=tags, **kwargs)


class VectorStoreRetriever(BaseRetriever):
    """Base Retriever class for VectorStore."""

    vectorstore: VectorStore
    """VectorStore to use for retrieval."""
    search_type: str = "similarity"
    """Type of search to perform."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "mmr",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_search_type(cls, values: dict) -> Any:
        """Validate search type.

        Args:
            values: Values to validate.

        Returns:
            Validated values.

        Raises:
            ValueError: If `search_type` is not one of the allowed search types.
            ValueError: If `score_threshold` is not specified with a float value(`0~1`)
        """
        search_type = values.get("search_type", "similarity")
        if search_type not in cls.allowed_search_types:
            msg = (
                f"search_type of {search_type} not allowed. Valid values are: "
                f"{cls.allowed_search_types}"
            )
            raise ValueError(msg)
        if search_type == "similarity_score_threshold":
            score_threshold = values.get("search_kwargs", {}).get("score_threshold")
            if (score_threshold is None) or (not isinstance(score_threshold, float)):
                msg = (
                    "`score_threshold` is not specified with a float value(0~1) "
                    "in `search_kwargs`."
                )
                raise ValueError(msg)
        return values

    def _get_ls_params(self, **kwargs: Any) -> LangSmithRetrieverParams:
        """Get standard params for tracing."""
        kwargs_ = self.search_kwargs | kwargs

        ls_params = super()._get_ls_params(**kwargs_)
        ls_params["ls_vector_store_provider"] = self.vectorstore.__class__.__name__

        if self.vectorstore.embeddings:
            ls_params["ls_embedding_provider"] = (
                self.vectorstore.embeddings.__class__.__name__
            )
        elif hasattr(self.vectorstore, "embedding") and isinstance(
            self.vectorstore.embedding, Embeddings
        ):
            ls_params["ls_embedding_provider"] = (
                self.vectorstore.embedding.__class__.__name__
            )

        return ls_params

    @override
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        kwargs_ = self.search_kwargs | kwargs
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **kwargs_)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **kwargs_
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(query, **kwargs_)
        else:
            msg = f"search_type of {self.search_type} not allowed."
            raise ValueError(msg)
        return docs

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        kwargs_ = self.search_kwargs | kwargs
        if self.search_type == "similarity":
            docs = await self.vectorstore.asimilarity_search(query, **kwargs_)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query, **kwargs_
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **kwargs_
            )
        else:
            msg = f"search_type of {self.search_type} not allowed."
            raise ValueError(msg)
        return docs

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Add documents to the `VectorStore`.

        Args:
            documents: Documents to add to the `VectorStore`.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            List of IDs of the added texts.
        """
        return self.vectorstore.add_documents(documents, **kwargs)

    async def aadd_documents(
        self, documents: list[Document], **kwargs: Any
    ) -> list[str]:
        """Async add documents to the `VectorStore`.

        Args:
            documents: Documents to add to the `VectorStore`.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            List of IDs of the added texts.
        """
        return await self.vectorstore.aadd_documents(documents, **kwargs)
