from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.docstore.base import AddableMixin, Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore


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
    """`USearch` vector store.

    To use, you should have the ``usearch`` python package installed.
    """

    def __init__(
        self,
        embedding: Embeddings,
        index: Any,
        docstore: Docstore,
        ids: List[str],
    ):
        """Initialize with necessary components."""
        self.embedding = embedding
        self.index = index
        self.docstore = docstore
        self.ids = ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )

        embeddings = self.embedding.embed_documents(list(texts))
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        last_id = int(self.ids[-1]) + 1
        if ids is None:
            ids = np.array([str(last_id + id) for id, _ in enumerate(texts)])

        self.index.add(np.array(ids), np.array(embeddings))
        self.docstore.add(dict(zip(ids, documents)))
        self.ids.extend(ids)
        return ids.tolist()

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of documents most similar to the query with distance.
        """
        query_embedding = self.embedding.embed_query(query)
        matches = self.index.search(np.array(query_embedding), k)

        docs_with_scores: List[Tuple[Document, float]] = []
        for id, score in zip(matches.keys, matches.distances):
            doc = self.docstore.search(str(id))
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {id}, got {doc}")
            docs_with_scores.append((doc, score))

        return docs_with_scores

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        query_embedding = self.embedding.embed_query(query)
        matches = self.index.search(np.array(query_embedding), k)

        docs: List[Document] = []
        for id in matches.keys:
            doc = self.docstore.search(str(id))
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {id}, got {doc}")
            docs.append(doc)

        return docs

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[np.ndarray] = None,
        metric: str = "cos",
        **kwargs: Any,
    ) -> USearch:
        """Construct USearch wrapper from raw documents.
        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the USearch database
        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import USearch
                from langchain_community.embeddings import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                usearch = USearch.from_texts(texts, embeddings)
        """
        embeddings = embedding.embed_documents(texts)

        documents: List[Document] = []
        if ids is None:
            ids = np.array([str(id) for id, _ in enumerate(texts)])
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))

        docstore = InMemoryDocstore(dict(zip(ids, documents)))
        usearch = dependable_usearch_import()
        index = usearch.Index(ndim=len(embeddings[0]), metric=metric)
        index.add(np.array(ids), np.array(embeddings))
        return cls(embedding, index, docstore, ids.tolist())
