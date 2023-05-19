"""Milvus Retriever"""
from typing import Any, Dict, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores.milvus import Milvus

# TODO: Update to MilvusClient + Hybrid Search when available


class MilvusRetreiver(BaseRetriever):
    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = "LangChainCollection",
        connection_args: Optional[Dict[str, Any]] = None,
        consistency_level: str = "Session",
        search_params: Optional[dict] = None,
    ):
        self.store = Milvus(
            embedding_function,
            collection_name,
            connection_args,
            consistency_level,
        )
        self.retriever = self.store.as_retriever(search_kwargs={"param": search_params})

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> None:
        """Add text to the Milvus store

        Args:
            texts (List[str]): The text
            metadatas (List[dict]): Metadata dicts, must line up with existing store
        """
        self.store.add_texts(texts, metadatas)

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.retriever.get_relevant_documents(query)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
