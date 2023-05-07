"""Wrapper around ChromaDB embeddings platform."""
import os
import vecdb

from typing import Iterable, Optional, List, Any, TypeVar, Type

from langchain.vectorstores import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document

VST = TypeVar("VST", bound="VectorStore")


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
        self.dataset.insert(data=texts, metadata=metadatas, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return self.dataset.search(page_size=k, text=query, **kwargs)

    @classmethod
    def from_texts(
        cls: Type[VST],
        dataset_id: str,
        texts: List[str],
        embedding: Embeddings = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "VecDB":
        store = cls(dataset_id)
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store
