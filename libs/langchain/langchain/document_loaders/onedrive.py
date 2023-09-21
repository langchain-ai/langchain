"""Loads data from OneDrive"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base_o365 import (
    O365BaseLoader,
    _FileType,
)
from langchain.document_loaders.parsers.registry import get_parser
from langchain.pydantic_v1 import Field

if TYPE_CHECKING:
    from O365.drive import Drive, Folder

logger = logging.getLogger(__name__)


class OneDriveLoader(O365BaseLoader):
    """Load from `Microsoft OneDrive`."""

    drive_id: str = Field(...)
    """ The ID of the OneDrive drive to load data from."""
    folder_path: Optional[str] = None
    """ The path to the folder to load data from."""
    object_ids: Optional[List[str]] = None
    """ The IDs of the objects to load data from."""

    @property
    def _file_types(self) -> Sequence[_FileType]:
        """Return supported file types."""
        return _FileType.DOC, _FileType.DOCX, _FileType.PDF

    @property
    def _scopes(self) -> List[str]:
        """Return required scopes."""
        return ["offline_access", "Files.Read.All"]

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
        try:
            from O365.drive import Drive
        except ImportError:
            raise ImportError(
                "O365 package not found, please install it with `pip install o365`"
            )
        drive = self._auth().storage().get_drive(self.drive_id)
        if not isinstance(drive, Drive):
            raise ValueError(f"There isn't a Drive with id {self.drive_id}.")
        blob_parser = get_parser("default")
        if self.folder_path:
            folder = self._get_folder_from_path(drive)
            for blob in self._load_from_folder(folder):
                yield from blob_parser.lazy_parse(blob)
        if self.object_ids:
            for blob in self._load_from_object_ids(drive, self.object_ids):
                yield from blob_parser.lazy_parse(blob)

    def load(self) -> List[Document]:
        """Load all documents."""
        return list(self.lazy_load())
