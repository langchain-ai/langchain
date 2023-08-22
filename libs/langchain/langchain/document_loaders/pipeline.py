from __future__ import annotations

from typing import Iterator, List, Optional, Sequence

from langchain.document_loaders.base import BaseLoader
from langchain.schema import BaseDocumentTransformer, Document
from langchain.text_splitter import TextSplitter


class DocumentPipeline(BaseLoader):
    """A document pipeline that can be used to load documents.

    A simple document pipeline that composes a loader and a list of transformers.
    """

    def __init__(
        self,
        loader: BaseLoader,
        *,
        transformers: Sequence[BaseDocumentTransformer] = (),
    ) -> None:
        """Initialize the document pipeline.

        Args:
            loader: The loader to use for loading the documents.
            transformers: The transformers to use for transforming the documents.
        """
        self.loader = loader
        self.transformers = transformers

    def lazy_load(self) -> Iterator[Document]:
        """Fetch the data from the data selector."""
        try:
            documents = self.loader.lazy_load()
        except NotImplementedError:
            documents = iter(self.loader.load())

        for document in documents:
            _docs = [document]
            for transformer in self.transformers:
                # List below is needed because of typing issue in langchain
                _docs = list(transformer.transform_documents(_docs))
            yield from _docs

    def load(self) -> List[Document]:
        """Fetch the data from the data selector."""
        raise NotImplementedError("Use lazy_load instead")

    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None
    ) -> List[Document]:
        """Fetch the data from the data selector."""
        raise NotImplementedError("Use lazy_load instead")
