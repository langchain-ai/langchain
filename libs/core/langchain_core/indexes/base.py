import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast, \
    AsyncIterable

from langchain_core.documents import Document
from langchain_core.indexes.types import DeleteResponse, UpsertResponse
from langchain_core.runnables import run_in_executor
from langchain_core.stores import BaseStore
from langchain_core.structured_query import StructuredQuery


class Index(BaseStore[str, Document], ABC):
    """Interface for a document index.

    Example:
        .. code-block:: python

            from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union, Iterable
            from uuid import uuid4

            from langchain_core.documents import Document
            from langchain_core.indexes import UpsertResponse, DeleteResponse, Index

            def uuid4_generator() -> Iterable[str]:
                while True:
                    yield str(uuid4())

            class DictIndex(Index):

                def __init__(self) -> None:
                    self.store = {}

                def upsert(
                    self,
                    documents: Iterable[Document],
                    *,
                    ids: Optional[Iterable[str]] = None,
                    **kwargs: Any,
                ) -> UpsertResponse:
                    ids = ids or uuid4_generator()
                    succeeded = []
                    for id_, doc in zip(ids, documents):
                        self.store[id_] = doc
                        succeeded.append(id_)
                    return UpsertResponse(succeeded=succeeded, failed=[])

                def delete_by_ids(self, ids: Iterable[str]) -> DeleteResponse:
                    succeeded = []
                    failed = []
                    for id_ in ids:
                        try:
                            del self.store[id_]
                        except Exception:
                            failed.append(id_)
                        else:
                            succeeded.append(id_)
                    return DeleteResponse(succeeded=succeeded, failed=failed)

                def lazy_get_by_ids(self, ids: Iterable[str]) -> Iterable[Document]:
                    for id in ids:
                        yield self.store[id]

                def yield_keys(
                    self, *, prefix: Optional[str] = None
                ) -> Union[Iterator[str]]:
                    prefix = prefix or ""
                    for key in self.store:
                        if key.startswith(prefix):
                            yield key
    """  # noqa: E501

    @abstractmethod
    def upsert(
        self,
        # TODO: Iterable or Iterator?
        documents: Iterable[Document],
        *,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> UpsertResponse:
        """Upsert documents to index."""

    async def aupsert(
        self,
        documents: AsyncIterable[Document],
        *,
        ids: Optional[AsyncIterable[str]] = None,
        **kwargs: Any,
    ) -> UpsertResponse:
        """Upsert documents to index."""
        # TODO: how to convert AsyncIterable -> Iterable
        return await run_in_executor(None, self.upsert, documents, ids=ids, **kwargs)

    @abstractmethod
    def delete_by_ids(self, ids: Iterable[str]) -> DeleteResponse:
        """Delete documents by id.

        Args:
            ids: IDs of the documents to delete.

        Returns:
           A dict ``{"succeeded": [...], "failed": [...]}`` with the IDs of the
           documents that were successfully deleted and the ones that failed to be
           deleted.
        """
    async def adelete_by_ids(self, ids: AsyncIterable[str]) -> DeleteResponse:
        """Upsert documents to index."""
        return await run_in_executor(None, self.delete_by_ids, ids)

    @abstractmethod
    def lazy_get_by_ids(self, ids: Iterable[str]) -> Iterable[Document]:
        """Lazily get documents by id.

        Args:
            ids: IDs of the documents to get.

        Yields:
           Document
        """

    async def alazy_get_by_ids(self, ids: AsyncIterable[str]) -> AsyncIterable[Document]:
        """Lazily get documents by id.

        Args:
            ids: IDs of the documents to get.

        Yields:
           Document
        """


    def get_by_ids(self, ids: Iterable[str]) -> List[Document]:
        """Get documents by id.

        Args:
            ids: IDs of the documents to get.

        Returns:
           A list of the requested Documents.
        """
        return list(self.lazy_get_by_ids(ids))

    def delete(
        self,
        *,
        ids: Optional[Iterable[str]] = None,
        filters: Union[
            StructuredQuery, Dict[str, Any], List[Dict[str, Any]], None
        ] = None,
        **kwargs: Any,
    ) -> DeleteResponse:
        """Default implementation only supports deletion by id.

        Override this method if the integration supports deletion by other parameters.

        Args:
            ids: IDs of the documents to delete. Must be specified.
            **kwargs: Other keywords args not supported by default. Will be ignored.

        Returns:
           A dict ``{"succeeded": [...], "failed": [...]}`` with the IDs of the
           documents that were successfully deleted and the ones that failed to be
           deleted.

        Raises:
            ValueError: if ids are not provided.
        """
        if ids is None:
            raise ValueError("Must provide ids to delete.")
        if filters:
            kwargs = {"filters": filters, **kwargs}
        if kwargs:
            warnings.warn(
                "Only deletion by ids is supported for this integration, all other "
                f"arguments are ignored. Received {kwargs=}"
            )
        return self.delete_by_ids(ids)

    def lazy_get(
        self,
        *,
        ids: Optional[Iterable[str]] = None,
        filters: Union[
            StructuredQuery, Dict[str, Any], List[Dict[str, Any]], None
        ] = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """Default implementation only supports get by id.

        Override this method if the integration supports get by other parameters.

        Args:
            ids: IDs of the documents to get. Must be specified.
            **kwargs: Other keywords args not supported by default. Will be ignored.

        Yields:
           Document.

        Raises:
            ValueError: if ids are not provided.
        """
        if ids is None:
            raise ValueError("Must provide ids to get.")
        if filters:
            kwargs = {"filters": filters, **kwargs}
        if kwargs:
            warnings.warn(
                "Only deletion by ids is supported for this integration, all other "
                f"arguments are ignored. Received {kwargs=}"
            )
        return self.lazy_get_by_ids(ids)

    def get(
        self,
        *,
        ids: Optional[Iterable[str]] = None,
        filters: Union[
            StructuredQuery, Dict[str, Any], List[Dict[str, Any]], None
        ] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Default implementation only supports get by id.

        Override this method if the integration supports get by other parameters.

        Args:
            ids: IDs of the documents to get. Must be specified.
            **kwargs: Other keywords args not supported by default. Will be ignored.

        Returns:
           A list of the requested Documents.

        Raises:
            ValueError: if ids are not provided.
        """
        return list(self.lazy_get(ids=ids, filters=filters, **kwargs))

    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        return cast(List[Optional[Document]], self.get_by_ids(keys))  # type: ignore[arg-type]

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        ids, documents = zip(*key_value_pairs)
        self.add(documents, ids=ids)

    def mdelete(self, keys: Sequence[str]) -> None:
        self.delete_by_ids(keys)  # type: ignore[arg-type]
