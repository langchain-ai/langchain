from langchain.document_loaders.base import BaseLoader
from typing import List, Optional
from langchain.docstore.document import Document
import io


class UnstructuredFileLoader(BaseLoader):
    """Loader that uses unstructured to load files."""

    def __init__(self, file_path: str):
        try:
            import unstructured
        except ImportError:
            raise ValueError(
                "unstructured package not found, please install it with "
                f"`pip install unstructured`"
            )
        self.file_path = file_path

    def load(self) -> List[Document]:
        from unstructured.partition.auto import partition

        elements = partition(filename=self.file_path)
        text = "\n\n".join([str(el) for el in elements])
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]


class UnstructuredTextLoader(BaseLoader):
    """Loader that uses unstructured to load text."""

    def __init__(self, text: str, metadata: Optional[dict] = None):
        try:
            import unstructured
        except ImportError:
            raise ValueError(
                "unstructured package not found, please install it with "
                f"`pip install unstructured`"
            )
        self.io = io.StringIO()
        self.io.write(text)
        self.metadata = metadata

    def load(self) -> List[Document]:
        from unstructured.partition.auto import partition

        elements = partition(file=self.io)
        text = "\n\n".join([str(el) for el in elements])
        return [Document(page_content=text, metadata=self.metadata)]
