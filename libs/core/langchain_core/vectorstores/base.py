"""**Vector store** stores embedded data and performs vector search.

One of the most common ways to store and search over unstructured data is to
embed it and store the resulting embedding vectors, and then query the store
and retrieve the data that are 'most similar' to the embedded query.

**Class hierarchy:**

.. code-block::

    VectorStore --> <name>  # Examples: Annoy, FAISS, Milvus

    BaseRetriever --> VectorStoreRetriever --> <name>Retriever  # Example: VespaRetriever

**Main helpers:**

.. code-block::

    Embeddings, Document
"""  # noqa: E501

from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from itertools import cycle
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from langchain_core._api import beta
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.config import run_in_executor
from langchain_core.utils.aiter import abatch_iterate
from langchain_core.utils.iter import batch_iterate

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )
    from langchain_core.documents import Document
    from langchain_core.indexing.base import UpsertResponse

logger = logging.getLogger(__name__)

VST = TypeVar("VST", bound="VectorStore")


class VectorStore(ABC):
    """Interface for vector store."""

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        # One of the kwargs should be `ids` which is a list of ids
        # associated with the texts.
        # This is not yet enforced in the type signature for backwards compatibility
        # with existing implementations.
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            **kwargs: vectorstore specific parameters.
                One of the kwargs should be `ids` which is a list of ids
                associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.

        Raises:
            ValueError: If the number of metadatas does not match the number of texts.
            ValueError: If the number of ids does not match the number of texts.
        """
        if type(self).upsert != VectorStore.upsert:
            # Import document in local scope to avoid circular imports
            from langchain_core.documents import Document

            # This condition is triggered if the subclass has provided
            # an implementation of the upsert method.
            # The existing add_texts
            texts_: Sequence[str] = (
                texts if isinstance(texts, (list, tuple)) else list(texts)
            )
            if metadatas and len(metadatas) != len(texts_):
                raise ValueError(
                    "The number of metadatas must match the number of texts."
                    "Got {len(metadatas)} metadatas and {len(texts_)} texts."
                )

            if "ids" in kwargs:
                ids = kwargs.pop("ids")
                if ids and len(ids) != len(texts_):
                    raise ValueError(
                        "The number of ids must match the number of texts."
                        "Got {len(ids)} ids and {len(texts_)} texts."
                    )
            else:
                ids = None

            metadatas_ = iter(metadatas) if metadatas else cycle([{}])
            ids_: Iterable[Union[str, None]] = ids if ids is not None else cycle([None])
            docs = [
                Document(page_content=text, metadata=metadata_, id=id_)
                for text, metadata_, id_ in zip(texts, metadatas_, ids_)
            ]
            upsert_response = self.upsert(docs, **kwargs)
            return upsert_response["succeeded"]
        raise NotImplementedError(
            f"`add_texts` has not been implemented for {self.__class__.__name__} "
        )

    # Developer guidelines:
    # Do not override streaming_upsert!
    @beta(message="Added in 0.2.11. The API is subject to change.")
    def streaming_upsert(
        self, items: Iterable[Document], /, batch_size: int, **kwargs: Any
    ) -> Iterator[UpsertResponse]:
        """Upsert documents in a streaming fashion.

        Args:
            items: Iterable of Documents to add to the vectorstore.
            batch_size: The size of each batch to upsert.
            **kwargs: Additional keyword arguments.
                kwargs should only include parameters that are common to all
                documents. (e.g., timeout for indexing, retry policy, etc.)
                kwargs should not include ids to avoid ambiguous semantics.
                Instead, the ID should be provided as part of the Document object.

        Yields:
            UpsertResponse: A response object that contains the list of IDs that were
            successfully added or updated in the vectorstore and the list of IDs that
            failed to be added or updated.

        .. versionadded:: 0.2.11
        """
        # The default implementation of this method breaks the input into
        # batches of size `batch_size` and calls the `upsert` method on each batch.
        # Subclasses can override this method to provide a more efficient
        # implementation.
        for item_batch in batch_iterate(batch_size, items):
            yield self.upsert(item_batch, **kwargs)

    # Please note that we've added a new method `upsert` instead of re-using the
    # existing `add_documents` method.
    # This was done to resolve potential ambiguities around the behavior of **kwargs
    # in existing add_documents / add_texts methods which could include per document
    # information (e.g., the `ids` parameter).
    # Over time the `add_documents` could be denoted as legacy and deprecated
    # in favor of the `upsert` method.
    @beta(message="Added in 0.2.11. The API is subject to change.")
    def upsert(self, items: Sequence[Document], /, **kwargs: Any) -> UpsertResponse:
        """Add or update documents in the vectorstore.

        The upsert functionality should utilize the ID field of the Document object
        if it is provided. If the ID is not provided, the upsert method is free
        to generate an ID for the document.

        When an ID is specified and the document already exists in the vectorstore,
        the upsert method should update the document with the new data. If the document
        does not exist, the upsert method should add the document to the vectorstore.

        Args:
            items: Sequence of Documents to add to the vectorstore.
            **kwargs: Additional keyword arguments.

        Returns:
            UpsertResponse: A response object that contains the list of IDs that were
            successfully added or updated in the vectorstore and the list of IDs that
            failed to be added or updated.

        .. versionadded:: 0.2.11
        """
        #  Developer guidelines:
        #
        #  Vectorstores implementations are free to extend `upsert` implementation
        #  to take in additional data per document.
        #
        #  This data **SHOULD NOT** be part of the **kwargs** parameter, instead
        #  sub-classes can use a Union type on `documents` to include additional
        #  supported formats for the input data stream.
        #
        #  For example,
        #
        #  .. code-block:: python
        #  from typing import TypedDict
        #
        #  class DocumentWithVector(TypedDict):
        #      document: Document
        #      vector: List[float]
        #
        #  def upsert(
        #          self,
        #          documents: Union[Iterable[Document], Iterable[DocumentWithVector]],
        #          /,
        #          **kwargs
        #  ) -> UpsertResponse:
        #      \"\"\"Add or update documents in the vectorstore.\"\"\"
        #      # Implementation should check if documents is an
        #      # iterable of DocumentWithVector or Document
        #      pass
        #
        #  Implementations that override upsert should include a new doc-string
        #  that explains the semantics of upsert and includes in code
        #  examples of how to insert using the alternate data formats.

        # The implementation does not delegate to the `add_texts` method or
        # the `add_documents` method by default since those implementations
        raise NotImplementedError(
            f"upsert has not been implemented for {self.__class__.__name__}"
        )

    @beta(message="Added in 0.2.11. The API is subject to change.")
    async def astreaming_upsert(
        self,
        items: AsyncIterable[Document],
        /,
        batch_size: int,
        **kwargs: Any,
    ) -> AsyncIterator[UpsertResponse]:
        """Upsert documents in a streaming fashion. Async version of streaming_upsert.

        Args:
            items: Iterable of Documents to add to the vectorstore.
            batch_size: The size of each batch to upsert.
            **kwargs: Additional keyword arguments.
                kwargs should only include parameters that are common to all
                documents. (e.g., timeout for indexing, retry policy, etc.)
                kwargs should not include ids to avoid ambiguous semantics.
                Instead the ID should be provided as part of the Document object.

        Yields:
            UpsertResponse: A response object that contains the list of IDs that were
            successfully added or updated in the vectorstore and the list of IDs that
            failed to be added or updated.

        .. versionadded:: 0.2.11
        """
        async for batch in abatch_iterate(batch_size, items):
            yield await self.aupsert(batch, **kwargs)

    @beta(message="Added in 0.2.11. The API is subject to change.")
    async def aupsert(
        self, items: Sequence[Document], /, **kwargs: Any
    ) -> UpsertResponse:
        """Add or update documents in the vectorstore. Async version of upsert.

        The upsert functionality should utilize the ID field of the Document object
        if it is provided. If the ID is not provided, the upsert method is free
        to generate an ID for the document.

        When an ID is specified and the document already exists in the vectorstore,
        the upsert method should update the document with the new data. If the document
        does not exist, the upsert method should add the document to the vectorstore.

        Args:
            items: Sequence of Documents to add to the vectorstore.
            **kwargs: Additional keyword arguments.

        Returns:
            UpsertResponse: A response object that contains the list of IDs that were
            successfully added or updated in the vectorstore and the list of IDs that
            failed to be added or updated.

        .. versionadded:: 0.2.11
        """
        #  Developer guidelines: See guidelines for the `upsert` method.
        # The implementation does not delegate to the `add_texts` method or
        # the `add_documents` method by default since those implementations
        return await run_in_executor(None, self.upsert, items, **kwargs)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        logger.debug(
            f"The embeddings property has not been "
            f"implemented for {self.__class__.__name__}"
        )
        return None

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete. If None, delete all. Default is None.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """

        raise NotImplementedError("delete method must be implemented by subclass.")

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
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
            ids: List of ids to retrieve.

        Returns:
            List of Documents.

        .. versionadded:: 0.2.11
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet support get_by_ids."
        )

    # Implementations should override this method to provide an async native version.
    async def aget_by_ids(self, ids: Sequence[str], /) -> List[Document]:
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
            ids: List of ids to retrieve.

        Returns:
            List of Documents.

        .. versionadded:: 0.2.11
        """
        return await run_in_executor(None, self.get_by_ids, ids)

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """Async delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete. If None, delete all. Default is None.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        return await run_in_executor(None, self.delete, ids, **kwargs)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Async run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
                Default is None.
            **kwargs: vectorstore specific parameters.

        Returns:
            List of ids from adding the texts into the vectorstore.

        Raises:
            ValueError: If the number of metadatas does not match the number of texts.
            ValueError: If the number of ids does not match the number of texts.
        """
        if type(self).aupsert != VectorStore.aupsert:
            # Import document in local scope to avoid circular imports
            from langchain_core.documents import Document

            # This condition is triggered if the subclass has provided
            # an implementation of the upsert method.
            # The existing add_texts
            texts_: Sequence[str] = (
                texts if isinstance(texts, (list, tuple)) else list(texts)
            )
            if metadatas and len(metadatas) != len(texts_):
                raise ValueError(
                    "The number of metadatas must match the number of texts."
                    "Got {len(metadatas)} metadatas and {len(texts_)} texts."
                )

            if "ids" in kwargs:
                ids = kwargs.pop("ids")
                if ids and len(ids) != len(texts_):
                    raise ValueError(
                        "The number of ids must match the number of texts."
                        "Got {len(ids)} ids and {len(texts_)} texts."
                    )
            else:
                ids = None

            metadatas_ = iter(metadatas) if metadatas else cycle([{}])
            ids_: Iterable[Union[str, None]] = ids if ids is not None else cycle([None])
            docs = [
                Document(page_content=text, metadata=metadata_, id=id_)
                for text, metadata_, id_ in zip(texts, metadatas_, ids_)
            ]
            upsert_response = await self.aupsert(docs, **kwargs)
            return upsert_response["succeeded"]
        return await run_in_executor(None, self.add_texts, texts, metadatas, **kwargs)

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add or update documents in the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            kwargs: Additional keyword arguments.
                if kwargs contains ids and documents contain ids,
                the ids in the kwargs will receive precedence.

        Returns:
            List of IDs of the added texts.

        Raises:
            ValueError: If the number of ids does not match the number of documents.
        """
        if type(self).upsert != VectorStore.upsert:
            from langchain_core.documents import Document

            if "ids" in kwargs:
                ids = kwargs.pop("ids")
                if ids and len(ids) != len(documents):
                    raise ValueError(
                        "The number of ids must match the number of documents. "
                        "Got {len(ids)} ids and {len(documents)} documents."
                    )

                documents_ = []

                for id_, document in zip(ids, documents):
                    doc_with_id = Document(
                        page_content=document.page_content,
                        metadata=document.metadata,
                        id=id_,
                    )
                    documents_.append(doc_with_id)
            else:
                documents_ = documents

            # If upsert has been implemented, we can use it to add documents
            return self.upsert(documents_, **kwargs)["succeeded"]

        # Code path that delegates to add_text for backwards compatibility
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Async run more documents through the embeddings and add to
        the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            kwargs: Additional keyword arguments.

        Returns:
            List of IDs of the added texts.

        Raises:
            ValueError: If the number of IDs does not match the number of documents.
        """
        # If either upsert or aupsert has been implemented, we delegate to them!
        if (
            type(self).aupsert != VectorStore.aupsert
            or type(self).upsert != VectorStore.upsert
        ):
            # If aupsert has been implemented, we can use it to add documents
            from langchain_core.documents import Document

            if "ids" in kwargs:
                ids = kwargs.pop("ids")
                if ids and len(ids) != len(documents):
                    raise ValueError(
                        "The number of ids must match the number of documents."
                        "Got {len(ids)} ids and {len(documents)} documents."
                    )

                documents_ = []

                for id_, document in zip(ids, documents):
                    doc_with_id = Document(
                        page_content=document.page_content,
                        metadata=document.metadata,
                        id=id_,
                    )
                    documents_.append(doc_with_id)
            else:
                documents_ = documents

            # The default implementation of aupsert delegates to upsert.
            upsert_response = await self.aupsert(documents_, **kwargs)
            return upsert_response["succeeded"]

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return await self.aadd_texts(texts, metadatas, **kwargs)

    def search(self, query: str, search_type: str, **kwargs: Any) -> List[Document]:
        """Return docs most similar to query using a specified search type.

        Args:
            query: Input text
            search_type: Type of search to perform. Can be "similarity",
                "mmr", or "similarity_score_threshold".
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.

        Raises:
            ValueError: If search_type is not one of "similarity",
                "mmr", or "similarity_score_threshold".
        """
        if search_type == "similarity":
            return self.similarity_search(query, **kwargs)
        elif search_type == "similarity_score_threshold":
            docs_and_similarities = self.similarity_search_with_relevance_scores(
                query, **kwargs
            )
            return [doc for doc, _ in docs_and_similarities]
        elif search_type == "mmr":
            return self.max_marginal_relevance_search(query, **kwargs)
        else:
            raise ValueError(
                f"search_type of {search_type} not allowed. Expected "
                "search_type to be 'similarity', 'similarity_score_threshold'"
                " or 'mmr'."
            )

    async def asearch(
        self, query: str, search_type: str, **kwargs: Any
    ) -> List[Document]:
        """Async return docs most similar to query using a specified search type.

        Args:
            query: Input text.
            search_type: Type of search to perform. Can be "similarity",
                "mmr", or "similarity_score_threshold".
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.

        Raises:
            ValueError: If search_type is not one of "similarity",
                "mmr", or "similarity_score_threshold".
        """
        if search_type == "similarity":
            return await self.asimilarity_search(query, **kwargs)
        elif search_type == "similarity_score_threshold":
            docs_and_similarities = await self.asimilarity_search_with_relevance_scores(
                query, **kwargs
            )
            return [doc for doc, _ in docs_and_similarities]
        elif search_type == "mmr":
            return await self.amax_marginal_relevance_search(query, **kwargs)
        else:
            raise ValueError(
                f"search_type of {search_type} not allowed. Expected "
                "search_type to be 'similarity', 'similarity_score_threshold' or 'mmr'."
            )

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.
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
        """
        The 'correct' relevance function
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
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance.

        Args:
            *args: Arguments to pass to the search method.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Tuples of (doc, similarity_score).
        """
        raise NotImplementedError

    async def asimilarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Async run similarity search with distance.

        Args:
            *args: Arguments to pass to the search method.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Tuples of (doc, similarity_score).
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
    ) -> List[Tuple[Document, float]]:
        """
        Default similarity search with relevance scores. Modify if necessary
        in subclass.
        Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]

    async def _asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Default similarity search with relevance scores. Modify if necessary
        in subclass.
        Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = await self.asimilarity_search_with_score(query, k, **kwargs)
        return [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs.

        Returns:
            List of Tuples of (doc, similarity_score).
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
                f" 0 and 1, got {docs_and_similarities}"
            )

        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                warnings.warn(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold}"
                )
        return docs_and_similarities

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Async return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)
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
                f" 0 and 1, got {docs_and_similarities}"
            )

        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                warnings.warn(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold}"
                )
        return docs_and_similarities

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Async return docs most similar to query.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.
        """

        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        return await run_in_executor(None, self.similarity_search, query, k=k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query vector.
        """
        raise NotImplementedError

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Async return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query vector.
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
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                Default is 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        raise NotImplementedError

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Async return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                Default is 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.

        Returns:
            List of Documents selected by maximal marginal relevance.
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
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                Default is 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        raise NotImplementedError

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Async return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                Default is 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents selected by maximal marginal relevance.
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
        cls: Type[VST],
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> VST:
        """Return VectorStore initialized from documents and embeddings.

        Args:
            documents: List of Documents to add to the vectorstore.
            embedding: Embedding function to use.
            **kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from documents and embeddings.
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)

    @classmethod
    async def afrom_documents(
        cls: Type[VST],
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> VST:
        """Async return VectorStore initialized from documents and embeddings.

        Args:
            documents: List of Documents to add to the vectorstore.
            embedding: Embedding function to use.
            **kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from documents and embeddings.
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return await cls.afrom_texts(texts, embedding, metadatas=metadatas, **kwargs)

    @classmethod
    @abstractmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        """Return VectorStore initialized from texts and embeddings.

        Args:
            texts: Texts to add to the vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
                Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from texts and embeddings.
        """

    @classmethod
    async def afrom_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        """Async return VectorStore initialized from texts and embeddings.

        Args:
            texts: Texts to add to the vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
                Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from texts and embeddings.
        """
        return await run_in_executor(
            None, cls.from_texts, texts, embedding, metadatas, **kwargs
        )

    def _get_retriever_tags(self) -> List[str]:
        """Get tags for retriever."""
        tags = [self.__class__.__name__]
        if self.embeddings:
            tags.append(self.embeddings.__class__.__name__)
        return tags

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        """Return VectorStoreRetriever initialized from this VectorStore.

        Args:
            **kwargs: Keyword arguments to pass to the search function.
                Can include:
                search_type (Optional[str]): Defines the type of search that
                    the Retriever should perform.
                    Can be "similarity" (default), "mmr", or
                    "similarity_score_threshold".
                search_kwargs (Optional[Dict]): Keyword arguments to pass to the
                    search function. Can include things like:
                        k: Amount of documents to return (Default: 4)
                        score_threshold: Minimum relevance threshold
                            for similarity_score_threshold
                        fetch_k: Amount of documents to pass to MMR algorithm
                            (Default: 20)
                        lambda_mult: Diversity of results returned by MMR;
                            1 for minimum diversity and 0 for maximum. (Default: 0.5)
                        filter: Filter by document metadata

        Returns:
            VectorStoreRetriever: Retriever class for VectorStore.

        Examples:

        .. code-block:: python

            # Retrieve more documents with higher diversity
            # Useful if your dataset has many similar documents
            docsearch.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 6, 'lambda_mult': 0.25}
            )

            # Fetch more documents for the MMR algorithm to consider
            # But only return the top 5
            docsearch.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 5, 'fetch_k': 50}
            )

            # Only retrieve documents that have a relevance score
            # Above a certain threshold
            docsearch.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.8}
            )

            # Only get the single most similar document from the dataset
            docsearch.as_retriever(search_kwargs={'k': 1})

            # Use a filter to only retrieve documents from a specific paper
            docsearch.as_retriever(
                search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
            )
        """
        tags = kwargs.pop("tags", None) or [] + self._get_retriever_tags()
        return VectorStoreRetriever(vectorstore=self, tags=tags, **kwargs)


class VectorStoreRetriever(BaseRetriever):
    """Base Retriever class for VectorStore."""

    vectorstore: VectorStore
    """VectorStore to use for retrieval."""
    search_type: str = "similarity"
    """Type of search to perform. Defaults to "similarity"."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "mmr",
    )

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type.

        Args:
            values: Values to validate.

        Returns:
            Values: Validated values.

        Raises:
            ValueError: If search_type is not one of the allowed search types.
            ValueError: If score_threshold is not specified with a float value(0~1)
        """
        search_type = values.get("search_type", "similarity")
        if search_type not in cls.allowed_search_types:
            raise ValueError(
                f"search_type of {search_type} not allowed. Valid values are: "
                f"{cls.allowed_search_types}"
            )
        if search_type == "similarity_score_threshold":
            score_threshold = values.get("search_kwargs", {}).get("score_threshold")
            if (score_threshold is None) or (not isinstance(score_threshold, float)):
                raise ValueError(
                    "`score_threshold` is not specified with a float value(0~1) "
                    "in `search_kwargs`."
                )
        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = await self.vectorstore.asimilarity_search(
                query, **self.search_kwargs
            )
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            List of IDs of the added texts.
        """
        return self.vectorstore.add_documents(documents, **kwargs)

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Async add documents to the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            List of IDs of the added texts.
        """
        return await self.vectorstore.aadd_documents(documents, **kwargs)
