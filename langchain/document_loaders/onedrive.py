"""Loader that loads data from OneDrive"""
from __future__ import annotations

import logging
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

from pydantic import BaseModel, BaseSettings, Field, FilePath, SecretStr

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.onedrive_file import OneDriveFileLoader

if TYPE_CHECKING:
    from O365 import Account
    from O365.drive import Drive, Folder

SCOPES = ["offline_access", "Files.Read.All"]
logger = logging.getLogger(__name__)


class _OneDriveSettings(BaseSettings):
    client_id: str = Field(..., env="O365_CLIENT_ID")
    client_secret: SecretStr = Field(..., env="O365_CLIENT_SECRET")

    class Config:
        env_prefix = ""
        case_sentive = False
        env_file = ".env"


class _OneDriveTokenStorage(BaseSettings):
    token_path: FilePath = Field(Path.home() / ".credentials" / "o365_token.txt")


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


class OneDriveLoader(BaseLoader, BaseModel):
    settings: _OneDriveSettings = Field(default_factory=_OneDriveSettings)
    drive_id: str = Field(...)
    folder_path: Optional[str] = None
    object_ids: Optional[List[str]] = None
    auth_with_token: bool = False

    def _auth(self) -> Type[Account]:
        """
        Authenticates the OneDrive API client using the specified
        authentication method and returns the Account object.

        Returns:
            Type[Account]: The authenticated Account object.
        """
        try:
            from O365 import FileSystemTokenBackend
        except ImportError:
            raise ValueError(
                "O365 package not found, please install it with `pip install o365`"
            )
        if self.auth_with_token:
            token_storage = _OneDriveTokenStorage()
            token_path = token_storage.token_path
            token_backend = FileSystemTokenBackend(
                token_path=token_path.parent, token_filename=token_path.name
            )
            account = Account(
                credentials=(
                    self.settings.client_id,
                    self.settings.client_secret.get_secret_value(),
                ),
                scopes=SCOPES,
                token_backend=token_backend,
                **{"raise_http_errors": False},
            )
        else:
            token_backend = FileSystemTokenBackend(
                token_path=Path.home() / ".credentials"
            )
            account = Account(
                credentials=(
                    self.settings.client_id,
                    self.settings.client_secret.get_secret_value(),
                ),
                scopes=SCOPES,
                token_backend=token_backend,
                **{"raise_http_errors": False},
            )
            # make the auth
            account.authenticate()
        return account

    def _get_folder_from_path(self, drive: Type[Drive]) -> Union[Folder, Drive]:
        """
        Returns the folder or drive object located at the
        specified path relative to the given drive.

        Args:
            drive (Type[Drive]): The root drive from which the folder path is relative.

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

    def _load_from_folder(self, folder: Type[Folder]) -> List[Document]:
        """
        Loads all supported document files from the specified folder
        and returns a list of Document objects.

        Args:
            folder (Type[Folder]): The folder object to load the documents from.

        Returns:
            List[Document]: A list of Document objects representing
            the loaded documents.

        """
        docs = []
        file_types = _SupportedFileTypes(file_types=["doc", "docx", "pdf"])
        file_mime_types = file_types.fetch_mime_types()
        items = folder.get_items()
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            for file in items:
                if file.is_file:
                    if file.mime_type in list(file_mime_types.values()):
                        loader = OneDriveFileLoader(file=file)
                        docs.extend(loader.load())
        return docs

    def _load_from_object_ids(self, drive: Type[Drive]) -> List[Document]:
        """
        Loads all supported document files from the specified OneDrive
        drive based on their object IDs and returns a list
        of Document objects.

        Args:
            drive (Type[Drive]): The OneDrive drive object
            to load the documents from.

        Returns:
            List[Document]: A list of Document objects representing
            the loaded documents.
        """
        docs = []
        file_types = _SupportedFileTypes(file_types=["doc", "docx", "pdf"])
        file_mime_types = file_types.fetch_mime_types()
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            for object_id in self.object_ids if self.object_ids else [""]:
                file = drive.get_item(object_id)
                if not file:
                    logging.warning(
                        "There isn't a file with "
                        f"object_id {object_id} in drive {drive}."
                    )
                    continue
                if file.is_file:
                    if file.mime_type in list(file_mime_types.values()):
                        loader = OneDriveFileLoader(file=file)
                        docs.extend(loader.load())
        return docs

    def load(self) -> List[Document]:
        """
        Loads all supported document files from the specified OneDrive drive a
        nd returns a list of Document objects.

        Returns:
            List[Document]: A list of Document objects
            representing the loaded documents.

        Raises:
            ValueError: If the specified drive ID
            does not correspond to a drive in the OneDrive storage.
        """
        account = self._auth()
        storage = account.storage()
        drive = storage.get_drive(self.drive_id)
        docs: List[Document] = []
        if not drive:
            raise ValueError(f"There isn't a drive with id {self.drive_id}.")
        if self.folder_path:
            folder = self._get_folder_from_path(drive=drive)
            docs.extend(self._load_from_folder(folder=folder))
        elif self.object_ids:
            docs.extend(self._load_from_object_ids(drive=drive))
        return docs
