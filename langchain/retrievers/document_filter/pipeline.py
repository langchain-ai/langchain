"""DocumentFilter that uses a pipeline of other filters."""
from typing import List

from langchain.retrievers.document_filter.base import (
    BaseDocumentFilter,
    RetrievedDocument,
)


class DocumentFilterPipeline(BaseDocumentFilter):
    """DocumentFilter that uses a pipeline of other filters."""

    filters: List[BaseDocumentFilter]
    """List of document filters that are chained together and run in sequence."""

    def filter(
        self, docs: List[RetrievedDocument], query: str
    ) -> List[RetrievedDocument]:
        """Filter down documents."""
        for _filter in self.filters:
            docs = _filter.filter(docs, query)
        return docs

    async def afilter(
        self, docs: List[RetrievedDocument], query: str
    ) -> List[RetrievedDocument]:
        """Filter down documents."""
        raise NotImplementedError
