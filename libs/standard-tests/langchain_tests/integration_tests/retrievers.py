from abc import abstractmethod
from typing import Type

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_tests.base import BaseStandardTests


class RetrieversIntegrationTests(BaseStandardTests):
    @property
    @abstractmethod
    def retriever_constructor(self) -> Type[BaseRetriever]: ...

    @property
    def retriever_constructor_params(self) -> dict:
        return {}

    @property
    @abstractmethod
    def retriever_query_example(self) -> str:
        """
        Returns a dictionary representing the "args" of an example retriever call.
        """
        ...

    @pytest.fixture
    def retriever(self) -> BaseRetriever:
        return self.retriever_constructor(**self.retriever_constructor_params)

    def test_k_constructor_param(self) -> None:
        """
        Test that the retriever constructor accepts a k parameter.
        """
        params = {
            k: v for k, v in self.retriever_constructor_params.items() if k != "k"
        }
        params_3 = {**params, "k": 3}
        retriever_3 = self.retriever_constructor(**params_3)
        result_3 = retriever_3.invoke(self.retriever_query_example)
        assert len(result_3) == 3
        assert all(isinstance(doc, Document) for doc in result_3)

        params_1 = {**params, "k": 1}
        retriever_1 = self.retriever_constructor(**params_1)
        result_1 = retriever_1.invoke(self.retriever_query_example)
        assert len(result_1) == 1
        assert all(isinstance(doc, Document) for doc in result_1)

    def test_invoke_with_k_kwarg(self, retriever: BaseRetriever) -> None:
        result_1 = retriever.invoke(self.retriever_query_example, k=1)
        assert len(result_1) == 1
        assert all(isinstance(doc, Document) for doc in result_1)

        result_3 = retriever.invoke(self.retriever_query_example, k=3)
        assert len(result_3) == 3
        assert all(isinstance(doc, Document) for doc in result_3)

    def test_invoke_returns_documents(self, retriever: BaseRetriever) -> None:
        """
        If invoked with the example params, the retriever should return a list of
        Documents.
        """
        result = retriever.invoke(self.retriever_query_example)

        assert isinstance(result, list)
        assert all(isinstance(doc, Document) for doc in result)

    async def test_ainvoke_returns_documents(self, retriever: BaseRetriever) -> None:
        """
        If ainvoked with the example params, the retriever should return a list of
        Documents.
        """
        result = await retriever.ainvoke(self.retriever_query_example)

        assert isinstance(result, list)
        assert all(isinstance(doc, Document) for doc in result)
