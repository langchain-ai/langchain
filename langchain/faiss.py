"""Wrapper around FAISS vector database."""
from typing import Callable, List

import faiss
import numpy as np

from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings


class FAISS:
    """Wrapper around FAISS vector database."""

    def __init__(
        self, embedding_function: Callable, index: faiss.IndexFlatL2, docstore: Docstore
    ):
        """Initialize with necessary components."""
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Return docs most similar to query."""
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
        """Construct FAISS wrapper from raw documents."""
        embeddings = embedding.embed_documents(texts)
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings, dtype=np.float32))
        documents = [Document(page_content=text) for text in texts]
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        return cls(embedding.embed_query, index, docstore)
