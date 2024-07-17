import uuid
from typing import Dict, Optional, Sequence, Any, List

from langchain_core.documents import Document
from langchain_core.indexing import UpsertResponse
from langchain_core.indexing.base import DocumentIndexer, DeleteResponse


class InMemoryIndexer(DocumentIndexer):
    """In memory sync indexer."""

    def __init__(self, *, store: Optional[Dict[str, Document]] = None) -> None:
        """An in memory implementation of a document indexer."""
        self.store = store if store is not None else {}

    def upsert(self, items: Sequence[Document], /, **kwargs: Any) -> UpsertResponse:
        """Upsert items into the indexer."""
        ok_ids = []

        for item in items:
            if item.id is None:
                id_ = uuid.uuid4()
                item_ = item.copy()
                item_.id = str(id_)
            else:
                item_ = item

            self.store[item_.id] = item_
            ok_ids.append(item_.id)

        return UpsertResponse(succeeded=ok_ids, failed=[])

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> DeleteResponse:
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

    def get(self, ids: Sequence[str], /, **kwargs: Any) -> List[Document]:
        """Get by ids."""
        found_documents = []

        for id_ in ids:
            if id_ in self.store:
                found_documents.append(self.store[id_])

        return found_documents
