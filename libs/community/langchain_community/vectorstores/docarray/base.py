from abc import ABC
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Type

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from docarray import BaseDoc
    from docarray.index.abstract import BaseDocIndex


def _check_docarray_import() -> None:
    try:
        import docarray

        da_version = docarray.__version__.split(".")
        if int(da_version[0]) == 0 and int(da_version[1]) <= 31:
            raise ImportError(
                f"To use the DocArrayHnswSearch VectorStore the docarray "
                f"version >=0.32.0 is expected, received: {docarray.__version__}."
                f"To upgrade, please run: `pip install -U docarray`."
            )
    except ImportError:
        raise ImportError(
            "Could not import docarray python package. "
            "Please install it with `pip install docarray`."
        )


class DocArrayIndex(VectorStore, ABC):
    """Base class for `DocArray` based vector stores."""

    def __init__(
        self,
        doc_index: "BaseDocIndex",
        embedding: Embeddings,
    ):
        """Initialize a vector store from DocArray's DocIndex."""
        self.doc_index = doc_index
        self.embedding = embedding

    @staticmethod
    def _get_doc_cls(**embeddings_params: Any) -> Type["BaseDoc"]:
        """Get docarray Document class describing the schema of DocIndex."""
        from docarray import BaseDoc
        from docarray.typing import NdArray

        class DocArrayDoc(BaseDoc):
            text: Optional[str] = Field(default=None, required=False)
            embedding: Optional[NdArray] = Field(**embeddings_params)
            metadata: Optional[dict] = Field(default=None, required=False)

        return DocArrayDoc

    @property
    def doc_cls(self) -> Type["BaseDoc"]:
        if self.doc_index._schema is None:
            raise ValueError("doc_index expected to have non-null _schema attribute.")
        return self.doc_index._schema

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed texts and add to the vector store.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        ids: List[str] = []
        embeddings = self.embedding.embed_documents(list(texts))
        for i, (t, e) in enumerate(zip(texts, embeddings)):
            m = metadatas[i] if metadatas else {}
            doc = self.doc_cls(text=t, embedding=e, metadata=m)
            self.doc_index.index([doc])
            ids.append(str(doc.id))

        return ids

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of documents most similar to the query text and
            cosine distance in float for each.
            Lower score represents more similarity.
        """
        query_embedding = self.embedding.embed_query(query)
        query_doc = self.doc_cls(embedding=query_embedding)  # type: ignore
        docs, scores = self.doc_index.find(query_doc, search_field="embedding", limit=k)

        result = [
            (Document(page_content=doc.text, metadata=doc.metadata), score)
            for doc, score in zip(docs, scores)
        ]
        return result

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        results = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in results]

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and relevance scores, normalized on a scale from 0 to 1.

        0 is dissimilar, 1 is most similar.
        """
        raise NotImplementedError()

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query vector.
        """

        query_doc = self.doc_cls(embedding=embedding)  # type: ignore
        docs = self.doc_index.find(
            query_doc, search_field="embedding", limit=k
        ).documents

        result = [
            Document(page_content=doc.text, metadata=doc.metadata) for doc in docs
        ]
        return result

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        query_embedding = self.embedding.embed_query(query)
        query_doc = self.doc_cls(embedding=query_embedding)  # type: ignore

        docs = self.doc_index.find(
            query_doc, search_field="embedding", limit=fetch_k
        ).documents

        mmr_selected = maximal_marginal_relevance(
            np.array(query_embedding), docs.embedding, k=k
        )
        results = [
            Document(page_content=docs[idx].text, metadata=docs[idx].metadata)
            for idx in mmr_selected
        ]
        return results
