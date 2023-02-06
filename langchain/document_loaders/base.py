"""Base loader class."""

from abc import ABC, abstractmethod
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


class BaseLoader(ABC):
    """Base loader class."""

    @abstractmethod
    def load(self) -> List[Document]:
        """Load data into document objects."""

    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None
    ) -> List[Document]:
        """Load documents and split into chunks."""
        if text_splitter is None:
            _text_splitter: TextSplitter = RecursiveCharacterTextSplitter()
        else:
            _text_splitter = text_splitter
        docs = self.load()
        return _text_splitter.split_documents(docs)
