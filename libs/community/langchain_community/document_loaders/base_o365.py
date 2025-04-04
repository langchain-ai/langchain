"""Base class for all loaders that uses O365 Package"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
import tempfile
import urllib
from abc import abstractmethod
from datetime import datetime
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Union

from pydantic import (
    BaseModel,
    Field,
    FilePath,
    PrivateAttr,
    SecretStr,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from langchain_community.document_loaders.base import BaseBlobParser, BaseLoader
from langchain_community.document_loaders.blob_loaders.file_system import (
    FileSystemBlobLoader,
)
from langchain_community.document_loaders.blob_loaders.schema import Blob
from langchain_community.document_loaders.parsers.generic import MimeTypeBasedParser
from langchain_community.document_loaders.parsers.registry import get_parser

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


def fetch_mime_types(file_types: Sequence[str]) -> Dict[str, str]:
    """Fetch the mime types for the specified file types."""
    mime_types_mapping = {}
    for ext in file_types:
        mime_type, _ = mimetypes.guess_type(f"file.{ext}")
        if mime_type:
            mime_types_mapping[ext] = mime_type
        else:
            raise ValueError(f"Unknown mimetype of extension {ext}")
    return mime_types_mapping


def fetch_extensions(mime_types: Sequence[str]) -> Dict[str, str]:
    """Fetch the mime types for the specified file types."""
    mime_types_mapping = {}
    for mime_type in mime_types:
        ext = mimetypes.guess_extension(mime_type)
        if ext:
            mime_types_mapping[ext[1:]] = mime_type  # ignore leading `.`
        else:
            raise ValueError(f"Unknown mimetype {mime_type}")
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
    modified_since: Optional[datetime] = None
    """Only fetch documents modified since given datetime. The datetime object
    must be timezone aware."""
    handlers: Optional[Dict[str, Any]] = {}
    """
    Provide custom handlers for MimeTypeBasedParser.

    Pass a dictionary mapping either file extensions (like "doc", "pdf", etc.) 
    or MIME types (like "application/pdf", "text/plain", etc.) to parsers. 
    Note that you must use either file extensions or MIME types exclusively and 
    cannot mix them.

    Do not include the leading dot for file extensions.
    
    Example using file extensions:
    ```python
        handlers = {
            "doc": MsWordParser(),
            "pdf": PDFMinerParser(),
            "txt": TextParser()
        }
    ```
    
    Example using MIME types:
    ```python
        handlers = {
            "application/msword": MsWordParser(),
            "application/pdf": PDFMinerParser(),
            "text/plain": TextParser()
        }
    ```
    """

    _blob_parser: BaseBlobParser = PrivateAttr()
    _file_types: Sequence[str] = PrivateAttr()
    _mime_types: Dict[str, str] = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.handlers:
            handler_keys = list(self.handlers.keys())
            try:
                # assume handlers.keys() are file extensions
                self._mime_types = fetch_mime_types(handler_keys)
                self._file_types = list(set(handler_keys))
                mime_handlers = {
                    self._mime_types[extension]: handler
                    for extension, handler in self.handlers.items()
                }
            except ValueError:
                try:
                    # assume handlers.keys() are mime types
                    self._mime_types = fetch_extensions(handler_keys)
                    self._file_types = list(set(self._mime_types.keys()))
                    mime_handlers = self.handlers
                except ValueError:
                    raise ValueError(
                        "`handlers` keys must be either file extensions or mimetypes.\n"
                        f"{handler_keys} could not be interpreted as either.\n"
                        "File extensions and mimetypes cannot mix. "
                        "Use either one or the other"
                    )

            self._blob_parser = MimeTypeBasedParser(
                handlers=mime_handlers, fallback_parser=None
            )
        else:
            self._blob_parser = get_parser("default")
            if not isinstance(self._blob_parser, MimeTypeBasedParser):
                raise TypeError(
                    'get_parser("default) was supposed to return MimeTypeBasedParser.'
                    f"It returned {type(self._blob_parser)}"
                )
            self._mime_types = fetch_extensions(list(self._blob_parser.handlers.keys()))

    @property
    def _fetch_mime_types(self) -> Dict[str, str]:
        """Return a dict of supported file types to corresponding mime types."""
        return self._mime_types

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
                        if (not self.modified_since) or (
                            file.modified > self.modified_since
                        ):
                            source = file.web_url
                            if re.search(
                                r"Doc.aspx\?sourcedoc=.*file=([^&]+)", file.web_url
                            ):
                                source = (
                                    file._parent.web_url
                                    + "/"
                                    + urllib.parse.quote(file.name)
                                )
                            file.download(to_path=temp_dir, chunk_size=self.chunk_size)
                            metadata_dict[file.name] = {
                                "source": source,
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
                        source = file.web_url
                        if re.search(
                            r"Doc.aspx\?sourcedoc=.*file=([^&]+)", file.web_url
                        ):
                            source = (
                                file._parent.web_url
                                + "/"
                                + urllib.parse.quote(file.name)
                            )
                        file.download(to_path=temp_dir, chunk_size=self.chunk_size)
                        metadata_dict[file.name] = {
                            "source": source,
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
