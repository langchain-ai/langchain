from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing_extensions import override

from langchain_tests.integration_tests import RetrieversIntegrationTests


class ParrotRetriever(BaseRetriever):
    parrot_name: str
    k: int = 3

    @override
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        k = kwargs.get("k", self.k)
        return [Document(page_content=f"{self.parrot_name} says: {query}")] * k


class TestParrotRetrieverIntegration(RetrieversIntegrationTests):
    @override
    @property
    def retriever_constructor(self) -> type[ParrotRetriever]:
        return ParrotRetriever

    @override
    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return {"parrot_name": "Polly"}

    @override
    @property
    def retriever_query_example(self) -> str:
        return "parrot"
