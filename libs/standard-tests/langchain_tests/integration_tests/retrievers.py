from abc import abstractmethod
from typing import Type

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_tests.base import BaseStandardTests


class RetrieversIntegrationTests(BaseStandardTests):
    """
    Base class for retrievers integration tests.
    """

    @property
    @abstractmethod
    def retriever_constructor(self) -> Type[BaseRetriever]:
        """
        A BaseRetriever subclass to be tested.
        """
        ...

    @property
    def retriever_constructor_params(self) -> dict:
        """
        Returns a dictionary of parameters to pass to the retriever constructor.
        """
        return {}

    @property
    @abstractmethod
    def retriever_query_example(self) -> str:
        """
        Returns a str representing the "query" of an example retriever call.
        """
        ...

    @pytest.fixture
    def retriever(self) -> BaseRetriever:
        """
        :private:
        """
        return self.retriever_constructor(**self.retriever_constructor_params)

    def test_k_constructor_param(self) -> None:
        """
        Test that the retriever constructor accepts a k parameter, representing
        the number of documents to return.

        .. dropdown:: Troubleshooting

            If this test fails, either the retriever constructor does not accept a k
            parameter, or the retriever does not return the correct number of documents
            (`k`) when it is set.

            For example, a retriever like

            .. code-block:: python

                    MyRetriever(k=3).invoke("query")

            should return 3 documents when invoked with a query.
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
        """
        Test that the invoke method accepts a k parameter, representing the number of
        documents to return.

        .. dropdown:: Troubleshooting

            If this test fails, the retriever's invoke method does not accept a k
            parameter, or the retriever does not return the correct number of documents
            (`k`) when it is set.

            For example, a retriever like

            .. code-block:: python

                MyRetriever().invoke("query", k=3)

            should return 3 documents when invoked with a query.
        """
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

        .. dropdown:: Troubleshooting

            If this test fails, the retriever's invoke method does not return a list of
            `langchain_core.document.Document` objects. Please confirm that your
            `_get_relevant_documents` method returns a list of `Document` objects.
        """
        result = retriever.invoke(self.retriever_query_example)

        assert isinstance(result, list)
        assert all(isinstance(doc, Document) for doc in result)

    async def test_ainvoke_returns_documents(self, retriever: BaseRetriever) -> None:
        """
        If ainvoked with the example params, the retriever should return a list of
        Documents.

        See :meth:`test_invoke_returns_documents` for more information on
        troubleshooting.
        """
        result = await retriever.ainvoke(self.retriever_query_example)

        assert isinstance(result, list)
        assert all(isinstance(doc, Document) for doc in result)
