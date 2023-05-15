from pydantic import BaseModel
from typing import List, Iterator, Type, Mapping, Any

from langchain.document_loaders.base import BaseLoader
from langchain.schema import BaseRetriever, Document
from langchain.utilities.arxiv import ArxivAPIWrapper


class QuerySupportingLoader(BaseLoader, BaseModel):
    """A parameterized loader."""

    query: str


class ArxivLoader(QuerySupportingLoader):
    """Load documents from Arxiv.

    SHOULD LIVE WITH DOCUMENT LOADERS
    """

    arxiv_api_wrapper: ArxivAPIWrapper

    def load(self) -> List[Document]:
        """We should stop implementing this and instead implement lazy_load()"""
        return list(self.lazy_load())

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """A lazy loader for document content."""
        yield from self.arxiv_api_wrapper.load(self.query)


class DocumentLoaderRetriever(BaseRetriever):
    loader_cls: Type[QuerySupportingLoader]
    additional_kwargs: Mapping[str, Any]

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query."""
        loader = self.loader_cls(query=query)
        return loader.load()

    def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
