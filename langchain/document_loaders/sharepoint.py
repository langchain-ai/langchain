"""Loader that loads data from Sharepoint Document Library"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

from langchain.docstore.document import Document
from langchain.document_loaders.base_o365 import O365BaseLoader
from langchain.document_loaders.parsers.registry import get_parser

SCOPES = ["sharepoint", "basic"]


class _FileType(str, Enum):
    DOC = "doc"
    DOCX = "docx"
    PDF = "pdf"


class _SupportedFileTypes(BaseModel):
    file_types: List[_FileType]

    def fetch_mime_types(self) -> Dict[str, str]:
        mime_types_mapping = {}
        for file_type in self.file_types:
            if file_type.value == "doc":
                mime_types_mapping[file_type.value] = "application/msword"
            elif file_type.value == "docx":
                mime_types_mapping[
                    file_type.value
                ] = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"  # noqa: E501
            elif file_type.value == "pdf":
                mime_types_mapping[file_type.value] = "application/pdf"
        return mime_types_mapping


class SharePointLoader(O365BaseLoader):
    document_library_id: str = Field(...)
    folder_path: Optional[str] = None
    object_ids: Optional[List[str]] = None
    auth_with_token: bool = False

    def _fetch_mime_types(self) -> Dict[str, str]:
        """This method will give you a Dictionary where the supported file types
        are the keys and their corresponding mime types are the values."""
        file_types = _SupportedFileTypes(file_types=["doc", "docx", "pdf"])
        file_mime_types = file_types.fetch_mime_types()
        return file_mime_types

    def lazy_load(self) -> Iterator[Document]:
        """Load documents lazily. Use this when working at a large scale."""
        try:
            from O365.drive import Drive, Folder
        except ImportError:
            raise ValueError(
                "O365 package not found, please install it with `pip install o365`"
            )
        account = self._auth(
            settings=self.settings, scopes=SCOPES, auth_with_token=self.auth_with_token
        )
        storage = account.storage()
        drive = storage.get_drive(self.document_library_id)
        blob_parser = get_parser("default")
        if not isinstance(drive, Drive):
            raise ValueError(f"There isn't a Drive with id {self.document_library_id}.")
        if self.folder_path:
            target_folder = drive.get_item_by_path(self.folder_path)
            if not isinstance(target_folder, Folder):
                raise ValueError(f"There isn't a folder with path {self.folder_path}.")
            for blob in self._load_from_folder(folder=target_folder):
                yield from blob_parser.lazy_parse(blob)
        if self.object_ids:
            for blob in self._load_from_object_ids(
                drive=drive, object_ids=self.object_ids
            ):
                yield from blob_parser.lazy_parse(blob)

    def load(self) -> List[Document]:
        """Load all documents."""
        return list(self.lazy_load())
