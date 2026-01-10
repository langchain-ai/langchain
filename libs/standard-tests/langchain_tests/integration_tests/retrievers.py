"""Integration tests for retrievers."""

from abc import abstractmethod
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_tests.base import BaseStandardTests


class RetrieversIntegrationTests(BaseStandardTests):
    """Base class for retrievers integration tests."""

    @property
    @abstractmethod
    def retriever_constructor(self) -> type[BaseRetriever]:
        """A `BaseRetriever` subclass to be tested."""
        ...

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        """Returns a dictionary of parameters to pass to the retriever constructor."""
        return {}

    @property
    @abstractmethod
    def retriever_query_example(self) -> str:
        """Returns a str representing the `query` of an example retriever call."""

    @property
    def num_results_arg_name(self) -> str:
        """Returns the name of the parameter for the number of results returned.

        Usually something like `k` or `top_k`.

        """
        return "k"

    @pytest.fixture
    def retriever(self) -> BaseRetriever:
        """Return retriever fixture."""
        return self.retriever_constructor(**self.retriever_constructor_params)

    def test_k_constructor_param(self) -> None:
        """Test the number of results constructor parameter.

        Test that the retriever constructor accepts a parameter representing
        the number of documents to return.

        By default, the parameter tested is named `k`, but it can be overridden by
        setting the `num_results_arg_name` property.

        !!! note
            If the retriever doesn't support configuring the number of results returned
            via the constructor, this test can be skipped using a pytest `xfail` on
            the test class:

            ```python
            @pytest.mark.xfail(
                reason="This retriever doesn't support setting "
                "the number of results via the constructor."
            )
            def test_k_constructor_param(self) -> None:
                raise NotImplementedError
            ```

        ??? note "Troubleshooting"

            If this test fails, the retriever constructor does not accept a number
            of results parameter, or the retriever does not return the correct number
            of documents ( of the one set in `num_results_arg_name`) when it is
            set.

            For example, a retriever like...

            ```python
            MyRetriever(k=3).invoke("query")
            ```

            ...should return 3 documents when invoked with a query.

        """
        params = {
            k: v
            for k, v in self.retriever_constructor_params.items()
            if k != self.num_results_arg_name
        }
        params_3 = {**params, self.num_results_arg_name: 3}
        retriever_3 = self.retriever_constructor(**params_3)
        result_3 = retriever_3.invoke(self.retriever_query_example)
        assert len(result_3) == 3
        assert all(isinstance(doc, Document) for doc in result_3)

        params_1 = {**params, self.num_results_arg_name: 1}
        retriever_1 = self.retriever_constructor(**params_1)
        result_1 = retriever_1.invoke(self.retriever_query_example)
        assert len(result_1) == 1
        assert all(isinstance(doc, Document) for doc in result_1)

    def test_invoke_with_k_kwarg(self, retriever: BaseRetriever) -> None:
        """Test the number of results parameter in `invoke`.

        Test that the invoke method accepts a parameter representing
        the number of documents to return.

        By default, the parameter is named, but it can be overridden by
        setting the `num_results_arg_name` property.

        !!! note
            If the retriever doesn't support configuring the number of results returned
            via the invoke method, this test can be skipped using a pytest `xfail` on
            the test class:

            ```python
            @pytest.mark.xfail(
                reason="This retriever doesn't support setting "
                "the number of results in the invoke method."
            )
            def test_invoke_with_k_kwarg(self) -> None:
                raise NotImplementedError
            ```

        ??? note "Troubleshooting"

            If this test fails, the retriever's invoke method does not accept a number
            of results parameter, or the retriever does not return the correct number
            of documents (`k` of the one set in `num_results_arg_name`) when it is
            set.

            For example, a retriever like...

            ```python
            MyRetriever().invoke("query", k=3)
            ```

            ...should return 3 documents when invoked with a query.

        """
        result_1 = retriever.invoke(
            self.retriever_query_example, None, **{self.num_results_arg_name: 1}
        )
        assert len(result_1) == 1
        assert all(isinstance(doc, Document) for doc in result_1)

        result_3 = retriever.invoke(
            self.retriever_query_example, None, **{self.num_results_arg_name: 3}
        )
        assert len(result_3) == 3
        assert all(isinstance(doc, Document) for doc in result_3)

    def test_invoke_returns_documents(self, retriever: BaseRetriever) -> None:
        """Test invoke returns documents.

        If invoked with the example params, the retriever should return a list of
        Documents.

        ??? note "Troubleshooting"

            If this test fails, the retriever's invoke method does not return a list of
            `Document` objects. Please confirm that your
            `_get_relevant_documents` method returns a list of `Document` objects.
        """
        result = retriever.invoke(self.retriever_query_example)

        assert isinstance(result, list)
        assert all(isinstance(doc, Document) for doc in result)

    async def test_ainvoke_returns_documents(self, retriever: BaseRetriever) -> None:
        """Test ainvoke returns documents.

        If `ainvoke`'d with the example params, the retriever should return a list of
        `Document` objects.

        See `test_invoke_returns_documents` for more information on
        troubleshooting.
        """
        result = await retriever.ainvoke(self.retriever_query_example)

        assert isinstance(result, list)
        assert all(isinstance(doc, Document) for doc in result)
