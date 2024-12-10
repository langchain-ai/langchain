from typing import Any, Type

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_tests.integration_tests import RetrieversIntegrationTests


class ParrotRetriever(BaseRetriever):
    parrot_name: str
    k: int = 3

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        k = kwargs.get("k", self.k)
        return [Document(page_content=f"{self.parrot_name} says: {query}")] * k


class TestParrotRetrieverIntegration(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[ParrotRetriever]:
        return ParrotRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"parrot_name": "Polly"}

    @property
    def retriever_query_example(self) -> str:
        return "parrot"
