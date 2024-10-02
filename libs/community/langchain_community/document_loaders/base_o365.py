"""Base class for all loaders that uses O365 Package"""

from __future__ import annotations

import logging
import os
import tempfile
from abc import abstractmethod
from enum import Enum
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence, Union

from pydantic import (
    BaseModel,
    Field,
    FilePath,
    SecretStr,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders.file_system import (
    FileSystemBlobLoader,
)
from langchain_community.document_loaders.blob_loaders.schema import Blob

if TYPE_CHECKING:
    from O365 import Account
    from O365.drive import Drive, Folder

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024 * 5


class _O365Settings(BaseSettings):
    client_id: str = Field(..., alias="O365_CLIENT_ID")
    client_secret: SecretStr = Field(..., alias="O365_CLIENT_SECRET")

    model_config = SettingsConfigDict(
        case_sensitive=False, env_file=".env", env_prefix="", extra="ignore"
    )


class _O365TokenStorage(BaseSettings):
    token_path: FilePath = Path.home() / ".credentials" / "o365_token.txt"


class _FileType(str, Enum):
    DOC = "doc"
    DOCX = "docx"
    PDF = "pdf"


def fetch_mime_types(file_types: Sequence[_FileType]) -> Dict[str, str]:
    """Fetch the mime types for the specified file types."""
    mime_types_mapping = {}
    for file_type in file_types:
        if file_type.value == "doc":
            mime_types_mapping[file_type.value] = "application/msword"
        elif file_type.value == "docx":
            mime_types_mapping[file_type.value] = (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"  # noqa: E501
            )
        elif file_type.value == "pdf":
            mime_types_mapping[file_type.value] = "application/pdf"
    return mime_types_mapping


class O365BaseLoader(BaseLoader, BaseModel):
    """Base class for all loaders that uses O365 Package"""

    settings: _O365Settings = Field(default_factory=_O365Settings)  # type: ignore[arg-type]
    """Settings for the Office365 API client."""
    auth_with_token: bool = False
    """Whether to authenticate with a token or not. Defaults to False."""
    chunk_size: Union[int, str] = CHUNK_SIZE
    """Number of bytes to retrieve from each api call to the server. int or 'auto'."""
    recursive: bool = False
    """Should the loader recursively load subfolders?"""

    @property
    @abstractmethod
    def _file_types(self) -> Sequence[_FileType]:
        """Return supported file types."""

    @property
    def _fetch_mime_types(self) -> Dict[str, str]:
        """Return a dict of supported file types to corresponding mime types."""
        return fetch_mime_types(self._file_types)

    @property
    @abstractmethod
    def _scopes(self) -> List[str]:
        """Return required scopes."""

    def _load_from_folder(self, folder: Folder) -> Iterable[Blob]:
        """Lazily load all files from a specified folder of the configured MIME type.

        Args:
            folder: The Folder instance from which the files are to be loaded. This
                Folder instance should represent a directory in a file system where the
                files are stored.

        Yields:
            An iterator that yields Blob instances, which are binary representations of
                the files loaded from the folder.
        """
        file_mime_types = self._fetch_mime_types
        items = folder.get_items()
        metadata_dict: Dict[str, Dict[str, Any]] = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.dirname(temp_dir), exist_ok=True)
            for file in items:
                if file.is_file:
                    if file.mime_type in list(file_mime_types.values()):
                        file.download(to_path=temp_dir, chunk_size=self.chunk_size)
                        metadata_dict[file.name] = {
                            "source": file.web_url,
                            "mime_type": file.mime_type,
                            "created": str(file.created),
                            "modified": str(file.modified),
                            "created_by": str(file.created_by),
                            "modified_by": str(file.modified_by),
                            "description": file.description,
                            "id": str(file.object_id),
                        }

            loader = FileSystemBlobLoader(path=temp_dir)
            for blob in loader.yield_blobs():
                if not isinstance(blob.path, PurePath):
                    raise NotImplementedError("Expected blob path to be a PurePath")
                if blob.path:
                    file_metadata_ = metadata_dict.get(str(blob.path.name), {})
                    blob.metadata.update(file_metadata_)
                yield blob
        if self.recursive:
            for subfolder in folder.get_child_folders():
                yield from self._load_from_folder(subfolder)

    def _load_from_object_ids(
        self, drive: Drive, object_ids: List[str]
    ) -> Iterable[Blob]:
        """Lazily load files specified by their object_ids from a drive.

        Load files into the system as binary large objects (Blobs) and return Iterable.

        Args:
            drive: The Drive instance from which the files are to be loaded. This Drive
                instance should represent a cloud storage service or similar storage
                system where the files are stored.
            object_ids: A list of object_id strings. Each object_id represents a unique
                identifier for a file in the drive.

        Yields:
            An iterator that yields Blob instances, which are binary representations of
            the files loaded from the drive using the specified object_ids.
        """
        file_mime_types = self._fetch_mime_types
        metadata_dict: Dict[str, Dict[str, Any]] = {}
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
                        file.download(to_path=temp_dir, chunk_size=self.chunk_size)
                        metadata_dict[file.name] = {
                            "source": file.web_url,
                            "mime_type": file.mime_type,
                            "created": file.created,
                            "modified": file.modified,
                            "created_by": str(file.created_by),
                            "modified_by": str(file.modified_by),
                            "description": file.description,
                            "id": str(file.object_id),
                        }

            loader = FileSystemBlobLoader(path=temp_dir)
            for blob in loader.yield_blobs():
                if not isinstance(blob.path, PurePath):
                    raise NotImplementedError("Expected blob path to be a PurePath")
                if blob.path:
                    file_metadata_ = metadata_dict.get(str(blob.path.name), {})
                    blob.metadata.update(file_metadata_)
                yield blob

    def _auth(self) -> Account:
        """Authenticates the OneDrive API client

        Returns:
            The authenticated Account object.
        """
        try:
            from O365 import Account, FileSystemTokenBackend
        except ImportError:
            raise ImportError(
                "O365 package not found, please install it with `pip install o365`"
            )
        if self.auth_with_token:
            token_storage = _O365TokenStorage()
            token_path = token_storage.token_path
            token_backend = FileSystemTokenBackend(
                token_path=token_path.parent, token_filename=token_path.name
            )
            account = Account(
                credentials=(
                    self.settings.client_id,
                    self.settings.client_secret.get_secret_value(),
                ),
                scopes=self._scopes,
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
                scopes=self._scopes,
                token_backend=token_backend,
                **{"raise_http_errors": False},
            )
            # make the auth
            account.authenticate()
        return account
