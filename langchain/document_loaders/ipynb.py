# pip install nbformat nbconvert
import nbformat
from nbconvert import MarkdownExporter
from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

class IpynbLoader(BaseLoader):
    """Load ipynb files and convert code to markdown"""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load from file path."""
        with open(self.file_path) as f:
            nb = nbformat.read(f, as_version=4)

        exporter = MarkdownExporter()
        output, resources = exporter.from_notebook_node(nb, resources={'output_files_dir': 'output'})
        metadata = {'source': self.file_path}
        return [Document(page_content=output, metadata=metadata)]

