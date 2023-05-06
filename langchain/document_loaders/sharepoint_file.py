from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, List

from pydantic import BaseModel

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader

if TYPE_CHECKING:
    from O365.drive import File

CHUNK_SIZE = 1024 * 1024 * 5


class SharePointFileLoader(BaseLoader, BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def load(self, file: File) -> List[Document]:  # type: ignore
        """Load Documents"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.name}"
            file.download(to_path=temp_dir, chunk_size=CHUNK_SIZE)
            loader = UnstructuredFileLoader(file_path)
            return loader.load()
