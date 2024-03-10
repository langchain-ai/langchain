"""Fake vectorstore to test retriaval chains."""
from itertools import cycle, zip_longest
from typing import Any, Iterable, List, Optional, Type

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore


class FakeVectorStore(VectorStore):
    """A fake VectroStore to test retriaval chains.

    It runs through a list of provided documents in an infinite cycle and
        returns them sequentially.
    """

    def __init__(self, documents: Optional[List[Document]] = None):
        self._documents: List[Document] = documents if documents else []
        self._iterator = cycle(self._documents)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Adds more texts to the fake vectorstore.

        The current iterator over documents would be refreshed.
        """
        start = len(self._documents)
        for text, metadata in zip_longest(texts, metadatas if metadatas else []):
            self._documents.append(
                Document(page_content=text, metadata=metadata if metadata else {})
            )
        end = len(self._documents)
        self._iterator = cycle(self._documents)
        return [str(i) for i in range(start, end)]

    @classmethod
    def from_texts(
        cls: Type["FakeVectorStore"],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "FakeVectorStore":
        """Returns VectorStore initialized from texts and embeddings."""
        vector_store = cls()
        vector_store.add_texts(texts=texts, metadatas=metadatas)
        return vector_store

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Returns docs most similar to query."""
        del query
        return [next(self._iterator) for _ in range(k)]
