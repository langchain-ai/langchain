from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
from typing import Tuple, Optional, List, Callable
import faiss

class FAISS(Docstore):

    def __init__(self, embedding_function: Callable, index: faiss.IndexFlatL2):
        self.embedding_function = embedding_function
        self.index = index


    def search(self, search: str) -> Tuple[str, Optional[Document]]:
        raise NotImplementedError("Direct search not implemented for FAISS.")

    def similarity_search(self, search: str) -> List[Document]:

