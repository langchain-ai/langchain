from typing import List

from langchain.schema import BaseRetriever, Document


class SequentialRetriever(BaseRetriever):
    """Test util that returns a sequence of documents"""

    sequential_responses: List[List[Document]]
    response_index: int = 0

    def _get_relevant_documents(  # type: ignore[override]
        self,
        query: str,
    ) -> List[Document]:
        if self.response_index >= len(self.sequential_responses):
            return []
        else:
            self.response_index += 1
            return self.sequential_responses[self.response_index - 1]

    async def _aget_relevant_documents(  # type: ignore[override]
        self,
        query: str,
    ) -> List[Document]:
        return self._get_relevant_documents(query)
