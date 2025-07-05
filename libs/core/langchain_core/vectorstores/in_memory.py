"""In-memory vector store."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
)

from typing_extensions import override

from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.load import dumpd, load
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import _cosine_similarity as cosine_similarity
from langchain_core.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from langchain_core.embeddings import Embeddings
    from langchain_core.indexing import UpsertResponse


class InMemoryVectorStore(VectorStore):
    """In-memory vector store implementation.

    Uses a dictionary, and computes cosine similarity for search using numpy.

    Setup:
        Install ``langchain-core``.

        .. code-block:: bash

            pip install -U langchain-core

    Key init args — indexing params:
        embedding_function: Embeddings
            Embedding function to use.

    Instantiate:
        .. code-block:: python

            from langchain_core.vectorstores import InMemoryVectorStore
            from langchain_openai import OpenAIEmbeddings

            vector_store = InMemoryVectorStore(OpenAIEmbeddings())

    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(id="1", page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(id="2", page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(id="3", page_content="i will be deleted :(")

            documents = [document_1, document_2, document_3]
            vector_store.add_documents(documents=documents)

    Inspect documents:
        .. code-block:: python

            top_n = 10
            for index, (id, doc) in enumerate(vector_store.store.items()):
                if index < top_n:
                    # docs have keys 'id', 'vector', 'text', 'metadata'
                    print(f"{id}: {doc['text']}")
                else:
                    break

    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: none

            * thud [{'bar': 'baz'}]

    Search with filter:
        .. code-block:: python

            def _filter_function(doc: Document) -> bool:
                return doc.metadata.get("bar") == "baz"

            results = vector_store.similarity_search(
                query="thud", k=1, filter=_filter_function
            )
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: none

            * thud [{'bar': 'baz'}]


    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(
                query="qux", k=1
            )
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: none

            * [SIM=0.832268] foo [{'baz': 'bar'}]

    Async:
        .. code-block:: python

            # add documents
            # await vector_store.aadd_documents(documents=documents)

            # delete documents
            # await vector_store.adelete(ids=["3"])

            # search
            # results = vector_store.asimilarity_search(query="thud", k=1)

            # search with score
            results = await vector_store.asimilarity_search_with_score(query="qux", k=1)
            for doc,score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: none

            * [SIM=0.832268] foo [{'baz': 'bar'}]

    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
            )
            retriever.invoke("thud")

        .. code-block:: none

            [Document(id='2', metadata={'bar': 'baz'}, page_content='thud')]

    """

    def __init__(self, embedding: Embeddings) -> None:
        """Initialize with the given embedding function.

        Args:
            embedding: embedding function to use.
        """
        # TODO: would be nice to change to
        # dict[str, Document] at some point (will be a breaking change)
        self.store: dict[str, dict[str, Any]] = {}
        self.embedding = embedding

    @property
    @override
    def embeddings(self) -> Embeddings:
        return self.embedding

    @override
    def delete(self, ids: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        if ids:
            for _id in ids:
                self.store.pop(_id, None)

    @override
    async def adelete(self, ids: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        self.delete(ids)

    @override
    def add_documents(
        self,
        documents: list[Document],
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add documents to the store."""
        texts = [doc.page_content for doc in documents]
        vectors = self.embedding.embed_documents(texts)

        if ids and len(ids) != len(texts):
            msg = (
                f"ids must be the same length as texts. "
                f"Got {len(ids)} ids and {len(texts)} texts."
            )
            raise ValueError(msg)

        id_iterator: Iterator[Optional[str]] = (
            iter(ids) if ids else iter(doc.id for doc in documents)
        )

        ids_ = []

        for doc, vector in zip(documents, vectors):
            doc_id = next(id_iterator)
            doc_id_ = doc_id or str(uuid.uuid4())
            ids_.append(doc_id_)
            self.store[doc_id_] = {
                "id": doc_id_,
                "vector": vector,
                "text": doc.page_content,
                "metadata": doc.metadata,
            }

        return ids_

    @override
    async def aadd_documents(
        self, documents: list[Document], ids: Optional[list[str]] = None, **kwargs: Any
    ) -> list[str]:
        """Add documents to the store."""
        texts = [doc.page_content for doc in documents]
        vectors = await self.embedding.aembed_documents(texts)

        if ids and len(ids) != len(texts):
            msg = (
                f"ids must be the same length as texts. "
                f"Got {len(ids)} ids and {len(texts)} texts."
            )
            raise ValueError(msg)

        id_iterator: Iterator[Optional[str]] = (
            iter(ids) if ids else iter(doc.id for doc in documents)
        )
        ids_: list[str] = []

        for doc, vector in zip(documents, vectors):
            doc_id = next(id_iterator)
            doc_id_ = doc_id or str(uuid.uuid4())
            ids_.append(doc_id_)
            self.store[doc_id_] = {
                "id": doc_id_,
                "vector": vector,
                "text": doc.page_content,
                "metadata": doc.metadata,
            }

        return ids_

    @override
    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their ids.

        Args:
            ids: The ids of the documents to get.

        Returns:
            A list of Document objects.
        """
        documents = []

        for doc_id in ids:
            doc = self.store.get(doc_id)
            if doc:
                documents.append(
                    Document(
                        id=doc["id"],
                        page_content=doc["text"],
                        metadata=doc["metadata"],
                    )
                )
        return documents

    @deprecated(
        alternative="VectorStore.add_documents",
        message=(
            "This was a beta API that was added in 0.2.11. It'll be removed in 0.3.0."
        ),
        since="0.2.29",
        removal="1.0",
    )
    def upsert(self, items: Sequence[Document], /, **_kwargs: Any) -> UpsertResponse:
        """[DEPRECATED] Upsert documents into the store.

        Args:
            items: The documents to upsert.

        Returns:
            The upsert response.
        """
        vectors = self.embedding.embed_documents([item.page_content for item in items])
        ids = []
        for item, vector in zip(items, vectors):
            doc_id = item.id or str(uuid.uuid4())
            ids.append(doc_id)
            self.store[doc_id] = {
                "id": doc_id,
                "vector": vector,
                "text": item.page_content,
                "metadata": item.metadata,
            }
        return {
            "succeeded": ids,
            "failed": [],
        }

    @deprecated(
        alternative="VectorStore.aadd_documents",
        message=(
            "This was a beta API that was added in 0.2.11. It'll be removed in 0.3.0."
        ),
        since="0.2.29",
        removal="1.0",
    )
    async def aupsert(
        self, items: Sequence[Document], /, **_kwargs: Any
    ) -> UpsertResponse:
        """[DEPRECATED] Upsert documents into the store.

        Args:
            items: The documents to upsert.

        Returns:
            The upsert response.
        """
        vectors = await self.embedding.aembed_documents(
            [item.page_content for item in items]
        )
        ids = []
        for item, vector in zip(items, vectors):
            doc_id = item.id or str(uuid.uuid4())
            ids.append(doc_id)
            self.store[doc_id] = {
                "id": doc_id,
                "vector": vector,
                "text": item.page_content,
                "metadata": item.metadata,
            }
        return {
            "succeeded": ids,
            "failed": [],
        }

    @override
    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Async get documents by their ids.

        Args:
            ids: The ids of the documents to get.

        Returns:
            A list of Document objects.
        """
        return self.get_by_ids(ids)

    def _similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Optional[Callable[[Document], bool]] = None,  # noqa: A002
    ) -> list[tuple[Document, float, list[float]]]:
        # get all docs with fixed order in list
        docs = list(self.store.values())

        if filter is not None:
            docs = [
                doc
                for doc in docs
                if filter(Document(page_content=doc["text"], metadata=doc["metadata"]))
            ]

        if not docs:
            return []

        similarity = cosine_similarity([embedding], [doc["vector"] for doc in docs])[0]

        # get the indices ordered by similarity score
        top_k_idx = similarity.argsort()[::-1][:k]

        return [
            (
                Document(
                    id=doc_dict["id"],
                    page_content=doc_dict["text"],
                    metadata=doc_dict["metadata"],
                ),
                float(similarity[idx].item()),
                doc_dict["vector"],
            )
            for idx in top_k_idx
            # Assign using walrus operator to avoid multiple lookups
            if (doc_dict := docs[idx])
        ]

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Optional[Callable[[Document], bool]] = None,  # noqa: A002
        **_kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search for the most similar documents to the given embedding.

        Args:
            embedding: The embedding to search for.
            k: The number of documents to return.
            filter: A function to filter the documents.

        Returns:
            A list of tuples of Document objects and their similarity scores.
        """
        return [
            (doc, similarity)
            for doc, similarity, _ in self._similarity_search_with_score_by_vector(
                embedding=embedding, k=k, filter=filter
            )
        ]

    @override
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding,
            k,
            **kwargs,
        )

    @override
    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        embedding = await self.embedding.aembed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding,
            k,
            **kwargs,
        )

    @override
    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    @override
    async def asimilarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return self.similarity_search_by_vector(embedding, k, **kwargs)

    @override
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, **kwargs)]

    @override
    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return [
            doc
            for doc, _ in await self.asimilarity_search_with_score(query, k, **kwargs)
        ]

    @override
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filter: Optional[Callable[[Document], bool]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        prefetch_hits = self._similarity_search_with_score_by_vector(
            embedding=embedding,
            k=fetch_k,
            filter=filter,
        )

        try:
            import numpy as np
        except ImportError as e:
            msg = (
                "numpy must be installed to use max_marginal_relevance_search "
                "pip install numpy"
            )
            raise ImportError(msg) from e

        mmr_chosen_indices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            [vector for _, _, vector in prefetch_hits],
            k=k,
            lambda_mult=lambda_mult,
        )
        return [prefetch_hits[idx][0] for idx in mmr_chosen_indices]

    @override
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        embedding_vector = self.embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    @override
    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        embedding_vector = await self.embedding.aembed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    @classmethod
    @override
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> InMemoryVectorStore:
        store = cls(
            embedding=embedding,
        )
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

    @classmethod
    @override
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> InMemoryVectorStore:
        store = cls(
            embedding=embedding,
        )
        await store.aadd_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

    @classmethod
    def load(
        cls, path: str, embedding: Embeddings, **kwargs: Any
    ) -> InMemoryVectorStore:
        """Load a vector store from a file.

        Args:
            path: The path to load the vector store from.
            embedding: The embedding to use.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            A VectorStore object.
        """
        path_: Path = Path(path)
        with path_.open("r") as f:
            store = load(json.load(f))
        vectorstore = cls(embedding=embedding, **kwargs)
        vectorstore.store = store
        return vectorstore

    def dump(self, path: str) -> None:
        """Dump the vector store to a file.

        Args:
            path: The path to dump the vector store to.
        """
        path_: Path = Path(path)
        path_.parent.mkdir(exist_ok=True, parents=True)
        with path_.open("w") as f:
            json.dump(dumpd(self.store), f, indent=2)
