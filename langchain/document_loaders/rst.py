import os
from typing import Iterator, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class PandocRSTLoader(BaseLoader):
    """Loads .rst file and convert it into a Document"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load data into document objects."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for document content."""
        try:
            import pypandoc
        except ImportError:
            raise ImportError(
                "Could not import pypandoc. "
                "Please install it with `pip install pypandoc`. "
            )

        # Check if file exists
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"No such file or directory: '{self.file_path}'")

        markdown = pypandoc.convert_file(self.file_path, "markdown", format="rst")

        metadata = {
            "source": self.file_path,
            "file_path": self.file_path,
            "file_name": os.path.basename(self.file_path),
            "file_type": ".rst",
        }

        doc = Document(page_content=markdown, metadata=metadata)

        yield doc
