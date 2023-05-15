"""Interface for retrieving metadata from the various document loaders."""
import os
import platform

from abc import ABC, abstractmethod
from datetime import datetime
from typing import IO, Optional

from langchain.document_loaders.utils.filetype import get_mime_type, get_extension


class BaseMetadata(ABC):
    """Sample metadata to be extracted from the different types of loaded documents."""

    @abstractmethod
    def source(self) -> str:
        """Get source of loaded document."""

    @abstractmethod
    def created_at(self) -> str:
        """Get the creation date of loaded document. In ISO-8601 formatted string."""

    @abstractmethod
    def updated_at(self) -> str:
        """Get the modified date of loaded document. In ISO-8601 formatted string."""

    @abstractmethod
    def mime_type(self) -> str:
        """Get mime type of loaded document."""

    @abstractmethod
    def extension(self) -> str:
        """Get file extension of loaded document."""


class UnstructuredMetadata(BaseMetadata):
    """Metadata that can be extracted from documents loaded in by Unstructured."""

    def __init__(
        self,
        file_IO: Optional[IO] = None,
        file_path: Optional[str] = None,
    ) -> None:
        self.file_IO = file_IO
        self.file_path = file_path

    def source(self) -> str:
        return self.file_path

    def created_at(self) -> str:
        """
        NOTE: Creation date is generally possible with Windows file systems. However,
        the same cannot be said for some UNIX file systems, although most modern ones do
        store creation date. Even so, the system call is not exposed in Python.
        (Anyone is welcome to write a wrapper around it). To avoid this trouble, for
        UNIX file systems, we simply fall back to modified date only.
        """
        if platform.system() == "Windows":
            created_at = os.path.getctime(self.file_path)

            # Convert to ISO8601 format
            created_at = datetime.utcfromtimestamp(created_at).isoformat()
        else:
            created_at = self.updated_at()

        return created_at

    def updated_at(self) -> str:
        updated_at = os.path.getmtime(self.file_path)

        # Convert to ISO8601 format
        return datetime.utcfromtimestamp(updated_at).isoformat()

    def mime_type(self) -> str:
        if self.file_IO:
            mime_type = get_mime_type(file_IO=self.file_IO)
        elif self.file_path:
            mime_type = get_mime_type(file_path=self.file_path)

        return mime_type

    def extension(self) -> str:
        return get_extension(self.file_path)
