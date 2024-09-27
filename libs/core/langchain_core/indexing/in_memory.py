import uuid
from collections.abc import Sequence
from typing import Any, Optional, cast

from pydantic import Field

from langchain_core._api import beta
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.indexing import UpsertResponse
from langchain_core.indexing.base import DeleteResponse, DocumentIndex


@beta(message="Introduced in version 0.2.29. Underlying abstraction subject to change.")
class InMemoryDocumentIndex(DocumentIndex):
    """In memory document index.

    This is an in-memory document index that stores documents in a dictionary.

    It provides a simple search API that returns documents by the number of
    counts the given query appears in the document.

    .. versionadded:: 0.2.29
    """

    store: dict[str, Document] = Field(default_factory=dict)
    top_k: int = 4

    def upsert(self, items: Sequence[Document], /, **kwargs: Any) -> UpsertResponse:
        """Upsert items into the index."""
        ok_ids = []

        for item in items:
            if item.id is None:
                id_ = str(uuid.uuid4())
                item_ = item.model_copy()
                item_.id = id_
            else:
                item_ = item
                id_ = item.id

            self.store[id_] = item_
            ok_ids.append(cast(str, item_.id))

        return UpsertResponse(succeeded=ok_ids, failed=[])

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> DeleteResponse:
        """Delete by ID."""
        if ids is None:
            raise ValueError("IDs must be provided for deletion")

        ok_ids = []

        for id_ in ids:
            if id_ in self.store:
                del self.store[id_]
                ok_ids.append(id_)

        return DeleteResponse(
            succeeded=ok_ids, num_deleted=len(ok_ids), num_failed=0, failed=[]
        )

    def get(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get by ids."""
        found_documents = []

        for id_ in ids:
            if id_ in self.store:
                found_documents.append(self.store[id_])

        return found_documents

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        counts_by_doc = []

        for document in self.store.values():
            count = document.page_content.count(query)
            counts_by_doc.append((document, count))

        counts_by_doc.sort(key=lambda x: x[1], reverse=True)
        return [doc.model_copy() for doc, count in counts_by_doc[: self.top_k]]
