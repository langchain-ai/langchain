"""Standard LangChain integration tests."""

from typing import Any

import pytest
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
    ToolsIntegrationTests,
)

from langchain_seltz import SeltzSearchResults, SeltzSearchRetriever


class TestSeltzSearchResultsStandard(ToolsIntegrationTests):
    """Standard integration tests for `SeltzSearchResults`."""

    @property
    def tool_constructor(self) -> type[BaseTool]:
        """Return the tool class to test."""
        return SeltzSearchResults

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        """Return example invoke params."""
        return {"query": "best time to visit japan", "max_documents": 5}


class TestSeltzSearchRetrieverStandard(RetrieversIntegrationTests):
    """Standard integration tests for `SeltzSearchRetriever`."""

    @property
    def retriever_constructor(self) -> type[BaseRetriever]:
        """Return the retriever class to test."""
        return SeltzSearchRetriever

    @property
    def retriever_query_example(self) -> str:
        """Return an example query string."""
        return "best time to visit japan"

    @pytest.mark.xfail(reason="Seltz API does not support overriding k at invoke time.")
    def test_invoke_with_k_kwarg(self, retriever: BaseRetriever) -> None:
        """Test invoke with k kwarg."""
        super().test_invoke_with_k_kwarg(retriever)
