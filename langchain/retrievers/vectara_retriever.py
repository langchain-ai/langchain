"""Milvus Retriever"""
from typing import Optional, List

from langchain.schema import BaseRetriever, Document
from langchain.vectorstores.vectara import Vectara

class VectaraRetriever(BaseRetriever):
    def __init__(
        self,
        store: Vectara,
        alpha: float = 0.025,       # called "lambda" in Vectara, but changed here to alpha since its a reserved word in python
        k: int = 5,
        filter: Optional[str] = None,
    ):
        self.store = store
        self.alpha = alpha
        self.k = k
        self.filter = filter

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> None:
        """Add text to the Vectara vectorstore

        Args:
            texts (List[str]): The text
            metadatas (List[dict]): Metadata dicts, must line up with existing store
        """
        self.store.add_texts(texts, metadatas)

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.store.similarity_search(query, k=self.k, alpha=self.alpha, filter=self.filter)
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
