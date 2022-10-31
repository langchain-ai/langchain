from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
from typing import Tuple, Optional, List, Callable
import faiss
import numpy as np

class FAISS(Docstore):

    def __init__(self, embedding_function: Callable, index: faiss.IndexFlatL2, docstore: Docstore, k: int):
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore
        self.k = k

    def similarity_search(self, search: str) -> List[Document]:
        embedding = self.embedding_function(search)
        _, I = self.index.search(np.array([embedding], dtype=np.float32), self.k)
        return [self.docstore.search(i) for i in I[0]]


