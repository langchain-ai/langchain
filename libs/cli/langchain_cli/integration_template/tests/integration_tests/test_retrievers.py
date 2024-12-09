from typing import Type

from __module_name__.retrievers import __ModuleName__Retriever
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)


class Test__ModuleName__Retriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[__ModuleName__Retriever]:
        """Get an empty vectorstore for unit tests."""
        return __ModuleName__Retriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"k": 2}

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a str representing the "query" of an example retriever call.
        """
        return "example query"
