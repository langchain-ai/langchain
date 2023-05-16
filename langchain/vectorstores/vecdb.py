"""Wrapper around VecDB embeddings platform."""
from __future__ import annotations

import os
import uuid
import vecdb

from typing import Iterable, Optional, List, Any, Type

from langchain.vectorstores import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document


class VecDB(VectorStore):
    def __init__(self, dataset_id: str):
        self.client = vecdb.Client(os.getenv("VECDB_API_KEY"))
        self.dataset = self.client.create_dataset(dataset_id)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        _ids = kwargs.pop("_ids")
        if _ids is None:
            _ids = [str(uuid.uuid4()) for _ in range(texts)]
            kwargs["ids"] = _ids
        self.dataset.insert(data=texts, metadata=metadatas, **kwargs)
        return _ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return self.dataset.search(page_size=k, text=query, **kwargs)["documents"]

    @classmethod
    def from_texts(
        cls: Type[VecDB],
        dataset_id: str,
        texts: List[str],
        embedding: Embeddings = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VecDB:
        store = cls(dataset_id)
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

    def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.delete_dataset(self.dataset.dataset_id)
