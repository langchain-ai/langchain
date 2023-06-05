"""Loader that loads data from OneDrive"""
from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, Field

from langchain.docstore.document import Document
from langchain.document_loaders.base_o365 import O365BaseLoader
from langchain.document_loaders.parsers.registry import get_parser

if TYPE_CHECKING:
    from O365.drive import Drive, Folder

SCOPES = ["offline_access", "Files.Read.All"]
logger = logging.getLogger(__name__)


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


class OneDriveLoader(O365BaseLoader):
    drive_id: str = Field(...)
    folder_path: Optional[str] = None
    object_ids: Optional[List[str]] = None
    auth_with_token: bool = False

    def _fetch_mime_types(self) -> Dict[str, str]:
        """This method will give you a Dictionary where the supported file types
        are the keys and their corresponding mime types are the values."""
        file_types = _SupportedFileTypes(file_types=["doc", "docx", "pdf"])
        file_mime_types = file_types.fetch_mime_types()
        return file_mime_types

    def _get_folder_from_path(self, drive: Drive) -> Union[Folder, Drive]:
        """
        Returns the folder or drive object located at the
        specified path relative to the given drive.

        Args:
            drive (Drive): The root drive from which the folder path is relative.

        Returns:
            Union[Folder, Drive]: The folder or drive object
            located at the specified path.

        Raises:
            FileNotFoundError: If the path does not exist.
        """

        subfolder_drive = drive
        if self.folder_path is None:
            return subfolder_drive

        subfolders = [f for f in self.folder_path.split("/") if f != ""]
        if len(subfolders) == 0:
            return subfolder_drive

        items = subfolder_drive.get_items()
        for subfolder in subfolders:
            try:
                subfolder_drive = list(filter(lambda x: subfolder in x.name, items))[0]
                items = subfolder_drive.get_items()
            except (IndexError, AttributeError):
                raise FileNotFoundError("Path {} not exist.".format(self.folder_path))
        return subfolder_drive

    def lazy_load(self) -> Iterator[Document]:
        """Load documents lazily. Use this when working at a large scale."""
        account = self._auth(
            settings=self.settings, scopes=SCOPES, auth_with_token=self.auth_with_token
        )
        storage = account.storage()
        drive = storage.get_drive(self.drive_id)
        blob_parser = get_parser("default")
        if not drive:
            raise ValueError(f"There isn't a Drive with id {self.drive_id}.")
        if self.folder_path:
            folder = self._get_folder_from_path(drive=drive)
            for blob in self._load_from_folder(folder=folder):
                yield from blob_parser.lazy_parse(blob)
        if self.object_ids:
            for blob in self._load_from_object_ids(drive=drive, object_ids=self.object_ids):
                yield from blob_parser.lazy_parse(blob)

    def load(self) -> List[Document]:
        """Load all documents."""
        return list(self.lazy_load())