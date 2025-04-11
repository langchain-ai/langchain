from langchain_core.retrievers import BaseRetriever, Document


class SequentialRetriever(BaseRetriever):
    """Test util that returns a sequence of documents"""

    sequential_responses: list[list[Document]]
    response_index: int = 0

    def _get_relevant_documents(  # type: ignore[override]
        self,
        query: str,
    ) -> list[Document]:
        if self.response_index >= len(self.sequential_responses):
            return []
        else:
            self.response_index += 1
            return self.sequential_responses[self.response_index - 1]

    async def _aget_relevant_documents(  # type: ignore[override]
        self,
        query: str,
    ) -> list[Document]:
        return self._get_relevant_documents(query)
