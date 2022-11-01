from typing import Callable, List, Optional, Tuple

import faiss
import numpy as np

from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings


class FAISS:
    def __init__(
        self, embedding_function: Callable, index: faiss.IndexFlatL2, docstore: Docstore
    ):
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore

    def similarity_search(self, search: str, k: int = 4) -> List[Document]:
        embedding = self.embedding_function(search)
        _, I = self.index.search(np.array([embedding], dtype=np.float32), k)
        return [self.docstore.search(str(i)) for i in I[0]]

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings):
        embeddings = embedding.embed_documents(texts)
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings, dtype=np.float32))
        documents = [Document(page_content=text) for text in texts]
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        return cls(embedding.embed_query, index, docstore)
