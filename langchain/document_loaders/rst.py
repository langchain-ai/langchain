import os
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class UnstructuredRSTLoader(BaseLoader):
    """Loads .rst file and convert it into a Document"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        try:
            from docutils import core
        except ImportError:
            raise ImportError(
                "Could not import 'core' from docutils Python package. "
                "Please install it with `pip install docutils`."
            )

        with open(self.file_path, "r") as file:
            data = file.read()

        html = core.publish_parts(data, writer_name="html")["html_body"]

        metadata = {
            "source": self.file_path,
            "file_path": self.file_path,
            "file_name": os.path.basename(self.file_path),
            "file_type": ".rst",
        }
        doc = Document(page_content=html, metadata=metadata)
        return [doc]
