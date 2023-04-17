""""""
from typing import Any, Dict, List, Tuple

from langchain.document_filter.base import PipelineFilter
from langchain.schema import Document
from langchain.text_splitter import TextSplitter


class SplitterDocumentFilter(PipelineFilter):
    splitter: TextSplitter
    """"""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def filter_pipeline(
        self, docs: List[Document], query: str, **kwargs: Any
    ) -> Tuple[List[Document], Dict]:
        """"""
        return self.splitter.split_documents(docs), {}
