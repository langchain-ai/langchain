from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_tests.integration_tests import RetrieversIntegrationTests


class ParrotRetriever(BaseRetriever):
    parrot_name: str
    k: int = 3

    def _get_relevant_documents(
        self, query: str, k: Optional[int] = None, **_: Any
    ) -> list[Document]:
        return [Document(page_content=f"{self.parrot_name} says: {query}")] * (
            k or self.k
        )


class TestParrotRetrieverIntegration(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> type[ParrotRetriever]:
        return ParrotRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"parrot_name": "Polly"}

    @property
    def retriever_query_example(self) -> str:
        return "parrot"


class ParrotRetrieverWithoutK(BaseRetriever):
    parrot_name: str

    def _get_relevant_documents(self, query: str, **_: Any) -> list[Document]:
        return [Document(page_content=f"{self.parrot_name} says: {query}")] * 3


class TestParrotRetrieverIntegrationWithoutK(RetrieversIntegrationTests):
    @property
    def has_k_init_arg(self) -> bool:
        return False

    @property
    def has_k_invoke_kwarg(self) -> bool:
        return False

    @property
    def retriever_constructor(self) -> type[ParrotRetrieverWithoutK]:
        return ParrotRetrieverWithoutK

    @property
    def retriever_constructor_params(self) -> dict:
        return {"parrot_name": "Polly"}

    @property
    def retriever_query_example(self) -> str:
        return "parrot"
