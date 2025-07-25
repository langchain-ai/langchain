from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing_extensions import override


class SequentialRetriever(BaseRetriever):
    """Test util that returns a sequence of documents."""

    sequential_responses: list[list[Document]]
    response_index: int = 0

    @override
    def _get_relevant_documents(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[Document]:
        if self.response_index >= len(self.sequential_responses):
            return []
        self.response_index += 1
        return self.sequential_responses[self.response_index - 1]

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[Document]:
        return self._get_relevant_documents(query)
