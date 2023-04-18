"""Wrapper for using TextSplitter as a BaseDocumentFilter."""
from typing import Any, List

from langchain.retrievers.document_filters.base import (
    BaseDocumentFilter,
    RetrievedDocument,
)
from langchain.text_splitter import TextSplitter


class SplitterDocumentFilter(BaseDocumentFilter):
    splitter: TextSplitter
    """TextSplitter to use for splitting retrieved documents."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def filter(
        self, docs: List[RetrievedDocument], query: str, **kwargs: Any
    ) -> List[RetrievedDocument]:
        """Split the retrieved documents."""
        reg_docs = [rdoc.to_document() for rdoc in docs]
        split_docs = self.splitter.split_documents(reg_docs)
        return [RetrievedDocument.from_document(doc) for doc in split_docs]

    async def afilter(
        self, docs: List[RetrievedDocument], query: str
    ) -> List[RetrievedDocument]:
        raise NotImplementedError
