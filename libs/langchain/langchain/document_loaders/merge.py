import inspect
from typing import Iterator, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class MergedDataLoader(BaseLoader):
    """Merge documents from a list of loaders"""

    def __init__(self, loaders: List, **kwargs):
        """Initialize with a list of loaders"""
        self.loaders = loaders

    def lazy_load(self, **kwargs) -> Iterator[Document]:
        """Lazy load docs from each individual loader."""
        for loader in self.loaders:
            param_keys = inspect.signature(loader.load).parameters.keys()
            param = {k: v for k, v in kwargs.items() if k in param_keys}
            # Check if lazy_load is implemented
            try:
                data = loader.lazy_load(**param)
            except NotImplementedError:
                data = loader.load(**param)
            for document in data:
                yield document

    def load(self, **kwargs) -> List[Document]:
        """Load docs."""
        return list(self.lazy_load(**kwargs))
