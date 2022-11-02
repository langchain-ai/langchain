"""Wrapper around FAISS vector database."""
from typing import Any, Callable, List

import numpy as np

from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings


class FAISS:
    """Wrapper around FAISS vector database.

    To use, you should have the ``faiss`` python package installed.

    Example:
        .. code-block:: python

            from langchain import FAISS
            faiss = FAISS(embedding_function, index, docstore)

    """

    def __init__(self, embedding_function: Callable, index: Any, docstore: Docstore):
        """Initialize with necessary components."""
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding_function(query)
        _, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        for i in indices[0]:
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            doc = self.docstore.search(str(i))
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {i}, got {doc}")
            docs.append(doc)
        return docs

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings) -> "FAISS":
        """Construct FAISS wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import FAISS
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                faiss = FAISS.from_texts(texts, embeddings)
        """
        try:
            import faiss
        except ImportError:
            raise ValueError(
                "Could not import faiss python package. "
                "Please it install it with `pip install faiss` "
                "or `pip install faiss-cpu` (depending on Python version)."
            )
        embeddings = embedding.embed_documents(texts)
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings, dtype=np.float32))
        documents = [Document(page_content=text) for text in texts]
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        return cls(embedding.embed_query, index, docstore)
