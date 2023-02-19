"""Loading logic for loading documents from a directory."""
from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.document_loaders.paged_pdf import PagedPDFLoader
from langchain.document_loaders.pdf import PDFLoader
from langchain.document_loaders.text import TextLoader

class DirectoryLoader(BaseLoader):
    """Loading logic for loading documents from a directory."""

    def __init__(self, path: str, glob: str = "**/*", loader: BaseLoader = 'unstructured'):
        """Initialize with path to directory and how to glob over it."""
        self.path = path
        self.glob = glob
        self.loader_dict = {'unstructured': UnstructuredFileLoader, 'paged_pdf': PagedPDFLoader, 'pdf': PDFLoader, 'text': TextLoader}
        self.loader = self.loader_dict[loader]

    def load(self) -> List[Document]:
        """Load documents."""
        p = Path(self.path)
        docs = []
        for i in p.glob(self.glob):
            if i.is_file():
                sub_docs = self.loader(str(i)).load()
                docs.extend(sub_docs)
        return docs
