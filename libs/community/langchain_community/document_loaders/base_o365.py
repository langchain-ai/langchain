"""Base class for all loaders that uses O365 Package"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Sequence,
    TypedDict,
    Union,
)

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
    BaseModel,
    BaseSettings,
    Field,
    FilePath,
    SecretStr,
)

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader

if TYPE_CHECKING:
    from O365 import Account
    from O365.drive import Drive, File, Folder

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024 * 5


class DriveItemMetadata(TypedDict):
    id: str
    parent_id: str
    mime_type: str
    modified_time: str
    title: str
    name: str
    folder_path: str
    source: str
    author: str
    size: int


class _O365Settings(BaseSettings):
    client_id: str = Field(..., env="O365_CLIENT_ID")
    client_secret: SecretStr = Field(..., env="O365_CLIENT_SECRET")

    class Config:
        env_prefix = ""
        case_sentive = False
        env_file = ".env"


class _O365TokenStorage(BaseSettings):
    token_path: FilePath = Path.home() / ".credentials" / "o365_token.txt"


class _FileType(str, Enum):
    DOC = "doc"
    DOCX = "docx"
    PDF = "pdf"
    PPTX = ".pptx"
    TXT = ".txt"
    XLSX = ".xlsx"


class _FileExtension(str, Enum):
    DOC = ".doc"
    DOCX = ".docx"
    PDF = ".pdf"
    PPTX = ".pptx"
    TXT = ".txt"
    XLSX = ".xlsx"


def fetch_mime_types(file_types: Sequence[_FileType]) -> Dict[str, str]:
    """Fetch the mime types for the specified file types."""
    mime_types_mapping = {}
    for file_type in file_types:
        if file_type.value == "doc":
            mime_types_mapping[file_type.value] = "application/msword"
        elif file_type.value == "docx":
            mime_types_mapping[
                file_type.value
            ] = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"  # noqa: E501
        elif file_type.value == "pdf":
            mime_types_mapping[file_type.value] = "application/pdf"
        elif file_type.value == _FileType.PPTX:
            mime_types_mapping[
                file_type.value
            ] = "application/vnd.openxmlformats-officedocument.presentationml.presentation"  # noqa: E501
        elif file_type.value == _FileType.TXT:
            mime_types_mapping[file_type.value] = "text/plain"
        elif file_type.value == _FileType.XLSX:
            mime_types_mapping[
                file_type.value
            ] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return mime_types_mapping


class O365BaseLoader(BaseLoader, BaseModel):
    """Base class for all loaders that uses O365 Package"""

    folder_path_pattern = re.compile("/drives/[a-zA-Z0-9-!_]*/root:(.*)")

    settings: _O365Settings = Field(default_factory=_O365Settings)
    """Settings for the Office365 API client."""
    token_storage: _O365TokenStorage = Field(default_factory=_O365TokenStorage)
    """Path for the token"""
    auth_with_token: bool = False
    """Whether to authenticate with a token or not. Defaults to False."""
    chunk_size: Union[int, str] = CHUNK_SIZE
    """Number of bytes to retrieve from each api call to the server. int or 'auto'."""
    account_kwargs: Dict[str, Any] = {"raise_http_errors": False}
    """The account kwargs to use."""
    file_loader_cls: Any = None
    """The file loader class to use."""
    file_loader_kwargs: Dict[str, Any] = {}
    """The file loader kwargs to use."""

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

    def _load_from_folder(
        self, folder: Folder, recursive: bool = False
    ) -> Iterable[Document]:
        """Lazily load all files from a specified folder of the configured MIME type.

        Args:
            folder: The Folder instance from which the files are to be loaded. This
                Folder instance should represent a directory in a file system where the
                files are stored.
            recursive: If true, will load all the sub-folders of the Folder.
                If False, will only load the files in the Folder

        Yields:
            An iterator that yields Documents.
        """
        file_mime_types = self._fetch_mime_types
        items = folder.get_items()
        file_mime_types_values = list(file_mime_types.values())
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.dirname(temp_dir), exist_ok=True)
            for drive_item in items:
                if drive_item.is_file:
                    if drive_item.mime_type in file_mime_types_values:
                        drive_item.download(
                            to_path=temp_dir, chunk_size=self.chunk_size
                        )
                        yield from self._load_from_file(drive_item, Path(temp_dir))
                if drive_item.is_folder and recursive:
                    yield from self._load_from_folder(drive_item, recursive)

    def _load_from_object_ids(
        self, drive: Drive, object_ids: List[str]
    ) -> Iterable[Document]:
        """Lazily load files specified by their object_ids from a drive.

        Args:
            drive: The Drive instance from which the files are to be loaded. This Drive
                instance should represent a cloud storage service or similar storage
                system where the files are stored.
            object_ids: A list of object_id strings. Each object_id represents a unique
                identifier for a file in the drive.

        Yields:
            An iterator that yields Document instances.
        """
        file_mime_types = self._fetch_mime_types
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
                        yield from self._load_from_file(file, temp_dir)

    def _load_metadata_from_file(self, file: File | Drive) -> DriveItemMetadata:
        """Return the metadata of a file

        Args:
            file: The file to extract the metadata from

        Return a dict of the file metadata
        """

        path_result = self.folder_path_pattern.search(file.parent_path)
        if path_result is None:
            logger.error("failed to find path for ", file.name)
            raise ValueError("failed to find path for ", file.name)

        return {
            "id": file.object_id,
            "parent_id": file.parent_id,
            "mime_type": file.mime_type,
            "name": file.name,
            "folder_path": path_result.group(1),
            "source": file.web_url,
            "modified_time": file.modified.strftime("%Y-%m-%d %H:%M:%S.%f%z"),
            "author": file.created_by,
            "size": file.size,
        }

    def _load_from_file(
        self, file: File | Drive, temp_dir_path: Path
    ) -> List[Document]:
        """Return the Documents of a Drive file

        Args:
            file: The Drive file
            temp_dir_path: The path in /tmp of the file to return

        Return a List of Document
        """
        file_path = temp_dir_path + "/" + file.name

        loader: BaseLoader
        if self.file_loader_cls is not None:
            loader = self.file_loader_cls(
                file=file, file_path=file_path, **self.file_loader_kwargs
            )
        else:
            if (
                file.extension
                == _FileExtension.DOC | file.extension
                == _FileExtension.DOCX
            ):
                loader = Docx2txtLoader(file_path)
            elif file.extension == _FileExtension.PDF:
                loader = UnstructuredPDFLoader(file_path)
            elif file.extension == _FileExtension.PPTX:
                loader = UnstructuredPowerPointLoader(file_path)
            elif file.extension == _FileExtension.TXT:
                loader = TextLoader(file_path, None, True)
            elif file.extension == _FileExtension.XLSX:
                loader = UnstructuredExcelLoader(file_path, mode="single")
            else:
                logger.error("Wrong file extension got:", file.extension)
                raise Exception(
                    "File extension do not match any excepted file type: `"
                    + file.extension
                    + "`"
                    + " for file: `"
                    + file.name
                    + "`"
                )
        docs = loader.load()
        if docs is None:
            raise Exception("Error when loading " + file.name)
        metadata = self._load_metadata_from_file(file)
        for doc in docs:
            doc.metadata = metadata
        return docs

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
            token_path = self.token_storage.token_path
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
                **self.account_kwargs,
            )
            # trigger a call to see if the token is still valid
            account.get_current_user()
        else:
            token_backend = FileSystemTokenBackend(
                token_path=self.token_storage.token_path
            )
            account = Account(
                credentials=(
                    self.settings.client_id,
                    self.settings.client_secret.get_secret_value(),
                ),
                scopes=self._scopes,
                token_backend=token_backend,
                **self.account_kwargs,
            )
            # make the auth
            account.authenticate()
        return account
