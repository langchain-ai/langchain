from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, List

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader

if TYPE_CHECKING:
    from O365.drive import File

CHUNK_SIZE = 1024 * 1024 * 5


class OneDriveFileLoader(BaseLoader, BaseModel):
    """Load a file from `Microsoft OneDrive`."""

    file: File = Field(...)
    """The file to load."""

    class Config:
        arbitrary_types_allowed = True

    def load(self) -> List[Document]:
        """Load Documents"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.file.name}"
            self.file.download(to_path=temp_dir, chunk_size=CHUNK_SIZE)
            loader = UnstructuredFileLoader(file_path)
            return loader.load()
