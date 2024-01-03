from typing import Iterator, List

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class MergedDataLoader(BaseLoader):
    """Merge documents from a list of loaders"""

    def __init__(self, loaders: List):
        """Initialize with a list of loaders"""
        self.loaders = loaders

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load docs from each individual loader."""
        for loader in self.loaders:
            # Check if lazy_load is implemented
            try:
                data = loader.lazy_load()
            except NotImplementedError:
                data = loader.load()
            for document in data:
                yield document

    def load(self) -> List[Document]:
        """Load docs."""
        return list(self.lazy_load())
