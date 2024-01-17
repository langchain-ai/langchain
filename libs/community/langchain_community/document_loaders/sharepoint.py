"""Loader that loads data from Sharepoint Document Library"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field

from langchain_community.document_loaders.base_o365 import (
    O365BaseLoader,
    _FileType,
)


class SharePointLoader(O365BaseLoader):
    """Load  from `SharePoint`."""

    document_library_id: str = Field(...)
    """ The ID of the SharePoint document library to load data from."""
    folder_path: Optional[str] = None
    """ The path to the folder to load data from."""
    object_ids: Optional[List[str]] = None
    """ The IDs of the objects to load data from."""
    recursive: bool = False
    """ configure whether to recursively fetch files from sub-folders."""

    @property
    def _file_types(self) -> Sequence[_FileType]:
        """Return supported file types."""
        return (
            _FileType.DOC,
            _FileType.DOCX,
            _FileType.PDF,
            _FileType.PPTX,
            _FileType.TXT,
            _FileType.XLSX,
        )

    @property
    def _scopes(self) -> List[str]:
        """Return required scopes."""
        return ["sharepoint", "basic"]

    def lazy_load(self) -> Iterator[Document]:
        """Load documents lazily. Use this when working at a large scale."""
        try:
            from O365.drive import Drive, Folder
        except ImportError:
            raise ImportError(
                "O365 package not found, please install it with `pip install o365`"
            )
        drive = self._auth().storage().get_drive(self.document_library_id)
        if not isinstance(drive, Drive):
            raise ValueError(f"There isn't a Drive with id {self.document_library_id}.")
        if self.folder_path:
            target_folder = drive.get_item_by_path(self.folder_path)
            if not isinstance(target_folder, Folder):
                raise ValueError(f"There isn't a folder with path {self.folder_path}.")
            yield from self._load_from_folder(target_folder, self.recursive)
        if self.object_ids:
            yield from self._load_from_object_ids(drive, self.object_ids)
        if not self.folder_path and not self.object_ids:
            target_folder = drive.get_root_folder()
            if not isinstance(target_folder, Folder):
                raise ValueError("There isn't a root folder in this Drive.")
            yield from self._load_from_folder(target_folder, self.recursive)

    def load(self) -> List[Document]:
        """Load all documents."""
        return list(self.lazy_load())
