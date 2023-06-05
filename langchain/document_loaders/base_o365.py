"""Base class for all loaders that uses O365 Package"""
from __future__ import annotations

import logging
import os
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Union

from pydantic import BaseModel, BaseSettings, Field, FilePath, SecretStr

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.blob_loaders.file_system import FileSystemBlobLoader
from langchain.document_loaders.blob_loaders.schema import Blob

if TYPE_CHECKING:
    from O365 import Account
    from O365.drive import Drive, Folder

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024 * 5


class _O365Settings(BaseSettings):
    client_id: str = Field(..., env="O365_CLIENT_ID")
    client_secret: SecretStr = Field(..., env="O365_CLIENT_SECRET")

    class Config:
        env_prefix = ""
        case_sentive = False
        env_file = ".env"


class _O365TokenStorage(BaseSettings):
    token_path: FilePath = Field(Path.home() / ".credentials" / "o365_token.txt")


class O365BaseLoader(BaseLoader, BaseModel):
    settings: _O365Settings = Field(default_factory=_O365Settings)

    @abstractmethod
    def _fetch_mime_types(self) -> Dict[str, str]:
        """This method will give you a Dictionary where the supported file types
        are the keys and their corresponding mime types are the values."""
        ...

    def _load_from_folder(self, folder: Folder) -> Iterable[Blob]:
        """
        Load all files from a specified folder which have a MIME type present in the MIME types the loader is looking for.
        Then, load them into the system as binary large objects (Blobs).

        Parameters
        ----------
        folder : Folder
            The Folder instance from which the files are to be loaded. This Folder instance should represent a directory
            in a file system where the files are stored.

        Yields
        -------
        Iterable[Blob]
            An iterator that yields Blob instances, which are binary representations of the files loaded from the folder.
        """
        file_mime_types = self._fetch_mime_types()
        items = folder.get_items()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.dirname(temp_dir), exist_ok=True)
            for file in items:
                if file.is_file:
                    if file.mime_type in list(file_mime_types.values()):
                        file.download(to_path=temp_dir, chunk_size=CHUNK_SIZE)
            loader = FileSystemBlobLoader(path=temp_dir)
            yield from loader.yield_blobs()

    def _load_from_object_ids(
        self, drive: Drive, object_ids: List[str]
    ) -> Iterable[Blob]:
        """
        Load files, specified by their object_ids, from a drive, and load them into the system as binary large objects (Blobs).

        Parameters
        ----------
        drive : Drive
            The Drive instance from which the files are to be loaded. This Drive instance should represent a cloud storage
            service or similar storage system where the files are stored.

        object_ids : List[str]
            A list of object_id strings. Each object_id represents a unique identifier for a file in the drive.

        Yields
        -------
        Iterable[Blob]
            An iterator that yields Blob instances, which are binary representations of the files loaded from the drive using
            the specified object_ids.
        """
        file_mime_types = self._fetch_mime_types()
        with tempfile.TemporaryDirectory() as temp_dir:
            for object_id in object_ids:
                file = drive.get_item(object_id)
                if not file:
                    logging.warning(
                        "There isn't a file with"
                        f"object_id {object_id} in drive {drive}."
                    )
                    continue
                if file.is_file:
                    if file.mime_type in list(file_mime_types.values()):
                        file.download(to_path=temp_dir, chunk_size=CHUNK_SIZE)
            loader = FileSystemBlobLoader(path=temp_dir)
            yield from loader.yield_blobs()

    def _auth(
        self, settings: _O365Settings, scopes: List[str], auth_with_token: bool = False
    ) -> Account:
        """
        Authenticates the OneDrive API client using the specified
        authentication method and returns the Account object.

        Returns:
            Account: The authenticated Account object.
        """
        try:
            from O365 import Account, FileSystemTokenBackend
        except ImportError:
            raise ValueError(
                "O365 package not found, please install it with `pip install o365`"
            )
        if auth_with_token:
            token_storage = _O365TokenStorage()
            token_path = token_storage.token_path
            token_backend = FileSystemTokenBackend(
                token_path=token_path.parent, token_filename=token_path.name
            )
            account = Account(
                credentials=(
                    settings.client_id,
                    settings.client_secret.get_secret_value(),
                ),
                scopes=scopes,
                token_backend=token_backend,
                **{"raise_http_errors": False},
            )
        else:
            token_backend = FileSystemTokenBackend(
                token_path=Path.home() / ".credentials"
            )
            account = Account(
                credentials=(
                    settings.client_id,
                    settings.client_secret.get_secret_value(),
                ),
                scopes=scopes,
                token_backend=token_backend,
                **{"raise_http_errors": False},
            )
            # make the auth
            account.authenticate()
        return account
