""""""
from typing import List, Optional

from langchain.experimental.retriever_eval.base import TestRetriever
from langchain.schema import Document
from langchain.text_splitter import TextSplitter
from langchain.vectorstores.base import VectorStoreRetriever


class VectorStoreTestRetriever(TestRetriever):
    base_retriever: VectorStoreRetriever
    text_splitter: Optional[TextSplitter] = None

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query):
        return self.base_retriever.get_relevant_documents(query)

    def aget_relevant_documents(self, query):
        raise NotImplementedError

    def _insert_documents(self, docs: List[Document]) -> None:
        self.base_retriever.vectorstore.add_documents(docs)

    def _transform_documents(self, docs: List[Document]) -> List[Document]:
        if self.text_splitter is None:
            return docs
        return self.text_splitter.split_documents(docs)
