"""Loader that loads word documents."""
import os
import tempfile
from abc import ABC

from typing import List, Optional
from urllib.parse import urlparse

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader


class BaseDOCXLoader(BaseLoader, ABC):
    """Base loader class for PDF files.
    Defaults to check for local file, but if the file is a web path, it will download it
    to a temporary file, and use that, then clean up the temporary file after completion
    """

    file_path: str
    web_path: Optional[str] = None

    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path
        if "~" in self.file_path:
            self.file_path = os.path.expanduser(self.file_path)

        # If the file is a web path, download it to a temporary file, and use that
        if not os.path.isfile(self.file_path) and self._is_valid_url(self.file_path):
            r = requests.get(self.file_path)

            if r.status_code != 200:
                raise ValueError(
                    "Check the url of your file; returned status code %s"
                    % r.status_code
                )

            self.web_path = self.file_path
            self.temp_file = tempfile.NamedTemporaryFile()
            self.temp_file.write(r.content)
            self.file_path = self.temp_file.name
        elif not os.path.isfile(self.file_path):
            raise ValueError("File path %s is not a valid file or url" % self.file_path)

    def __del__(self) -> None:
        if hasattr(self, "temp_file"):
            self.temp_file.close()

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if the url is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)


class Docx2txtLoader(BaseDOCXLoader):
    """Loads a DOCX with docx2txt and chunks at character level."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        try:
            import docx2txt  # noqa:F401
        except ImportError:
            raise ValueError(
                "docx2txt package not found, please install it with " "`pip install docx2txt`"
            )
        super().__init__(file_path)

    def load(self) -> List[Document]:
        """Load given path as single page."""
        import docx2txt
        return [
            Document(
                page_content=docx2txt.process(self.file_path),
                metadata={"source": self.file_path},
            )
        ]


class UnstructuredWordDocumentLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load word documents."""

    def _get_elements(self) -> List:
        from unstructured.__version__ import __version__ as __unstructured_version__
        from unstructured.file_utils.filetype import FileType, detect_filetype

        unstructured_version = tuple(
            [int(x) for x in __unstructured_version__.split(".")]
        )
        # NOTE(MthwRobinson) - magic will raise an import error if the libmagic
        # system dependency isn't installed. If it's not installed, we'll just
        # check the file extension
        try:
            import magic  # noqa: F401

            is_doc = detect_filetype(self.file_path) == FileType.DOC
        except ImportError:
            _, extension = os.path.splitext(self.file_path)
            is_doc = extension == ".doc"

        if is_doc and unstructured_version < (0, 4, 11):
            raise ValueError(
                f"You are on unstructured version {__unstructured_version__}. "
                "Partitioning .doc files is only supported in unstructured>=0.4.11. "
                "Please upgrade the unstructured package and try again."
            )

        if is_doc:
            from unstructured.partition.doc import partition_doc

            return partition_doc(filename=self.file_path, **self.unstructured_kwargs)
        else:
            from unstructured.partition.docx import partition_docx

            return partition_docx(filename=self.file_path, **self.unstructured_kwargs)
