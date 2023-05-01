import tempfile
from typing import List

try:
    from O365.drive import File
except ImportError:
    raise ValueError(
        "o365 package not found, please install it with `pip install o365`"
    )

from pydantic import BaseModel, Field

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader

CHUNK_SIZE = 1024 * 1024 * 5


class OneDriveFileLoader(BaseLoader, BaseModel):
    file: File = Field(...)

    class Config:
        arbitrary_types_allowed = True

    def load(self) -> List[Document]:
        """Load Documents"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.file.name}"
            self.file.download(to_path=temp_dir, chunk_size=CHUNK_SIZE)
            loader = UnstructuredFileLoader(file_path)
            return loader.load()
