import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

from langchain_core.documents import Document
from langchain_core.indexes.types import AddResponse, DeleteResponse
from langchain_core.stores import BaseStore
from langchain_core.structured_query import StructuredQuery


class Index(BaseStore[str, Document], ABC):
    """Interface for a document index.

    Example:
        .. code-block:: python

            from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union
            from uuid import uuid4

            from langchain_core.documents import Document
            from langchain_core.vectorstores_v2 import AddResponse, DeleteResponse, Index


            class DictIndex(Index):

                def __init__(self) -> None:
                    self.store = {}

                def add(
                    self,
                    documents: Sequence[Document],
                    *,
                    ids: Optional[Union[List[str], Tuple[str]]] = None,
                    **kwargs: Any,
                ) -> AddResponse:
                    ids = ids or [str(uuid4()) for _ in documents]
                    self.store.update(dict(zip(ids, documents)))
                    return AddResponse(succeeded=list(ids), failed=[])

                def delete_by_ids(self, ids: Union[List[str], Tuple[str]]) -> DeleteResponse:
                    succeeded = []
                    failed = []
                    for id in ids:
                        try:
                            del self.store[id]
                        except Exception:
                            failed.append(id)
                        else:
                            succeeded.append(id)
                    return DeleteResponse(succeeded=succeeded, failed=failed)

                def get_by_ids(self, ids: Union[List[str], Tuple[str]]) -> List[Document]:
                    return [self.store[id] for id in ids]

                def yield_keys(
                    self, *, prefix: Optional[str] = None
                ) -> Union[Iterator[str]]:
                    prefix = prefix or ""
                    for key in self.store:
                        if key.startswith(prefix):
                            yield key
    """  # noqa: E501

    @abstractmethod
    def add(
        self,
        documents: Sequence[Document],
        *,
        ids: Optional[Union[List[str], Tuple[str]]] = None,
        **kwargs: Any,
    ) -> AddResponse:
        """Add documents to index."""

    @abstractmethod
    def delete_by_ids(self, ids: Union[List[str], Tuple[str]]) -> DeleteResponse:
        """Delete documents by id.

        Args:
            ids: IDs of the documents to delete.

        Returns:
           A dict ``{"succeeded": [...], "failed": [...]}`` with the IDs of the
           documents that were successfully deleted and the ones that failed to be
           deleted.
        """

    @abstractmethod
    def get_by_ids(self, ids: Union[List[str], Tuple[str]]) -> List[Document]:
        """Get documents by id.

        Args:
            ids: IDs of the documents to get.

        Returns:
           A list of the requested Documents.
        """

    def delete(
        self,
        *,
        ids: Optional[Union[List[str], Tuple[str]]] = None,
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

    def get(
        self,
        *,
        ids: Optional[Union[List[str], Tuple[str]]] = None,
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
        if ids is None:
            raise ValueError("Must provide ids to get.")
        if filters:
            kwargs = {"filters": filters, **kwargs}
        if kwargs:
            warnings.warn(
                "Only deletion by ids is supported for this integration, all other "
                f"arguments are ignored. Received {kwargs=}"
            )
        return self.get_by_ids(ids)

    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        return cast(List[Optional[Document]], self.get_by_ids(keys))  # type: ignore[arg-type]

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        ids, documents = zip(*key_value_pairs)
        self.add(documents, ids=ids)

    def mdelete(self, keys: Sequence[str]) -> None:
        self.delete_by_ids(keys)  # type: ignore[arg-type]

    # QUESTION: do we need Index.update or Index.upsert? should Index.add just do that?
    # QUESTION: should we support lazy versions of operations?
