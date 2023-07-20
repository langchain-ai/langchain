"""Wrapper around USearch vector database."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from langchain.docstore.base import AddableMixin, Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import DistanceStrategy, maximal_marginal_relevance


def dependable_usearch_import() -> Any:
    """
    Import usearch if available, otherwise raise error.
    """
    try:
        import usearch.index
    except ImportError:
        raise ImportError(
            "Could not import usearch python package. "
            "Please install it with `pip install usearch` "
        )
    return usearch.index


class USearch(VectorStore):

    def __init__(
        self,
        embedding_function: Callable,
        index: Any,
        docstore: Docstore,
        ids: List[str]
    ):
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore
        self.ids = ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> List[str]:

        embeddings = [self.embedding_function(text) for text in texts]
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        last_id = int(self.ids[-1]) + 1
        if ids is None:
            ids = [str(last_id + id) for id in range(len(texts))]
        self.index.add(np.array(ids), np.array(embeddings))
        self.docstore.add(dict(zip(ids, documents)))
        self.ids.extend(ids)
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:

        query_embedding = self.embedding_function(query)
        matches, distances, count = self.index.search(np.array(query_embedding), k)

        docs = [self.docstore.search(str(id)) for id in matches]
        return docs

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> USearch:

        embeddings = embedding.embed_documents(texts)

        documents = []
        if ids is None:
            ids = [str(id) for id, _ in enumerate(texts)]
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))

        docstore = InMemoryDocstore(dict(zip(ids, documents)))

        usearch = dependable_usearch_import()
        index = usearch.Index(
            ndim=3,  # Define the number of dimensions in input vectors
            metric='cos',  # Choose 'l2sq', 'haversine' or other metric, default = 'ip'
            dtype='f32',  # Quantize to 'f16' or 'f8' if needed, default = 'f32'
            connectivity=16,  # How frequent should the connections in the graph be, optional
            expansion_add=128,  # Control the recall of indexing, optional
            expansion_search=64)

        index.add(np.array(ids), np.array(embeddings))
        return cls(
            embedding.embed_query,
            index,
            docstore,
            ids
        )
