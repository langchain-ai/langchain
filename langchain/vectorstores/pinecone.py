"""Wrapper around Pinecone vector database."""
from __future__ import annotations

from typing import Any, Iterable, List, Optional

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore


class Pinecone(VectorStore):
    """Wrapper around Pinecone vector database.

    To use, you should have the ``pinecone-client`` python package installed.

    Example:
        .. code-block:: python

            import pinecone
            from langchain.vectorstores import Pinecone
            from langchain.embeddings.openai import OpenAIEmbeddings

            index = pinecone.Index('example_index')
            # supports any Embedding from langchain.embeddings
            embeddings = OpenAIEmbeddings()
            vectorstore = Pinecone(index, embeddings, "text_key")
    """

    def __init__(
        self,
        index: Any,
        embedding: Embeddings,
        text_key: str,
    ):
        """Initialize with Pinecone client."""
        try:
            import pinecone
        except ImportError:
            raise ValueError(
                "Could not import pinecone python package. "
                "Please it install it with `pip install pinecone-client`."
            )
        if not isinstance(index, pinecone.index.Index):
            raise ValueError(
                f"client should be an instance of pinecone.index.Index, got {type(client)}"
            )
        self._index = index
        self._embedding = embedding
        self._text_key = text_key

    def add_texts(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None
    ) -> None:
        """Not implemented for Pinecone yet."""
        raise NotImplementedError("Pinecone does not currently support `add_texts`.")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Look up similar documents in pinecone."""
        query_obj = self._embedding.embed_query(query)
        docs = []
        for res in self._index.query([query_obj], top_k=5, include_metadata=True)[
            "matches"
        ]:
            metadata = res["metadata"]
            text = metadata.pop(self._text_key)
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VectorStore:
        """Not implemented for Pinecone yet."""
        raise NotImplementedError("Pinecone does not currently support `from_texts`.")
