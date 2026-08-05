"""A vector store stores embedded data and performs vector search.

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
        metadatas: list[dict[str, Any]] | None = None,
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
        metadatas: list[dict[str, Any]] | None = None,
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
        # Check if subclass has implemented aadd_documents 
        if type(self).aadd_documents != VectorStore.aadd_documents:
            # Subclass HAS overridden aadd_documents, so proceed normally
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
            # Call subclass implementation using run_in_executor
            return await run_in_executor(None, self.aadd_documents, docs, **kwargs)
        else:
            # No override - raise NotImplementedError like the sync version does
            msg = f"`aadd_documents` has not been implemented for {self.__class__.__name__} "
            raise NotImplementedError(msg)

    async def aadd_documents(
        self, documents: list[Document], ids: list[str] | None = None, **kwargs: Any
    ) -> list[str]:
        """Async run more documents through the embeddings and add to the `VectorStore`.

        Args:
            documents: List of documents to add to the `VectorStore`.
            ids: Optional list of IDs associated with the documents.
            **kwargs: `VectorStore` specific parameters.

        Returns:
            List of IDs from adding the documents into the `VectorStore`.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        msg = f"`aadd_documents` has not been implemented for {self.__class__.__name__} "
        raise NotImplementedError(msg)
