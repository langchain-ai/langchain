from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class FakeParrotRetriever(BaseRetriever):
    """Test util that parrots the query back as documents."""

    def _get_relevant_documents(  # type: ignore[override]
        self,
        query: str,
    ) -> List[Document]:
        return [Document(page_content=query)]

    async def _aget_relevant_documents(  # type: ignore[override]
        self,
        query: str,
    ) -> List[Document]:
        return [Document(page_content=query)]
