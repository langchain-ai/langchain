from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class RoamLoader(BaseLoader):
    """Load `Roam` files from a directory."""

    def __init__(self, path: Union[str, Path]):
        """Initialize with a path."""
        self.file_path = path

    def load(self) -> List[Document]:
        """Load documents."""
        ps = list(Path(self.file_path).glob("**/*.md"))
        docs = []
        for p in ps:
            with open(p) as f:
                text = f.read()
            metadata = {"source": str(p)}
            docs.append(Document(page_content=text, metadata=metadata))
        return docs
