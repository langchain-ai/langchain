from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from langchain.schema import Document


class BaseDocumentFilter(BaseModel, ABC):
    @abstractmethod
    def filter(self, docs: List[Document], query: str) -> List[Document]:
        """Filter down documents."""

    @abstractmethod
    def afilter(self, docs: List[Document], query: str) -> List[Document]:
        """Filter down documents."""


class PipelineFilter(BaseDocumentFilter, ABC):
    """"""

    def filter(self, docs: List[Document], query: str) -> List[Document]:
        """Filter down documents."""
        docs, _ = self.filter_pipeline(docs, query)
        return docs

    @abstractmethod
    def filter_pipeline(
        self, docs: List[Document], query: str, **kwargs: Any
    ) -> Tuple[List[Document], Dict]:
        """"""

    def afilter(self, docs: List[Document], query: str) -> List[Document]:
        """Filter down documents."""
        raise NotImplementedError


class DocumentFilterPipeline(BaseDocumentFilter):
    filters: List[PipelineFilter]
    """"""

    def filter(self, docs: List[Document], query: str) -> List[Document]:
        """Filter down documents."""
        kwargs: Dict = {}
        for _filter in self.filters:
            docs, kwargs = _filter.filter_pipeline(docs, query, **kwargs)
        return docs

    def afilter(self, docs: List[Document], query: str) -> List[Document]:
        """Filter down documents."""
        raise NotImplementedError
