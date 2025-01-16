"""Loads word documents."""

import os
import tempfile
from abc import ABC
from pathlib import Path
from typing import Any, List, Union
from urllib.parse import urlparse

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import (
    UnstructuredFileLoader,
    validate_unstructured_version,
)


class Docx2txtLoader(BaseLoader, ABC):
    """Load `DOCX` file using `docx2txt` and chunks at character level.

    Defaults to check for local file, but if the file is a web path, it will download it
    to a temporary file, and use that, then clean up the temporary file after completion
    """

    def __init__(self, file_path: Union[str, Path]):
        """Initialize with file path."""
        self.file_path = str(file_path)
        self.original_file_path = self.file_path
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

    def load(self) -> List[Document]:
        """Load given path as single page."""
        import docx2txt

        return [
            Document(
                page_content=docx2txt.process(self.file_path),
                metadata={"source": self.original_file_path},
            )
        ]

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if the url is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)


class UnstructuredWordDocumentLoader(UnstructuredFileLoader):
    """Load `Microsoft Word` file using `Unstructured`.

    Works with both .docx and .doc files.
    You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader

    loader = UnstructuredWordDocumentLoader(
        "example.docx", mode="elements", strategy="fast",
    )
    docs = loader.load()

    References
    ----------
    https://unstructured-io.github.io/unstructured/bricks.html#partition-docx
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        mode: str = "single",
        **unstructured_kwargs: Any,
    ):
        """

        Args:
            file_path: The path to the Word file to load.
            mode: The mode to use when loading the file. Can be one of "single",
                "multi", or "all". Default is "single".
            **unstructured_kwargs: Any kwargs to pass to the unstructured.
        """
        file_path = str(file_path)
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.file_utils.filetype import FileType, detect_filetype

        # NOTE(MthwRobinson) - magic will raise an import error if the libmagic
        # system dependency isn't installed. If it's not installed, we'll just
        # check the file extension
        try:
            import magic  # noqa: F401

            is_doc = detect_filetype(self.file_path) == FileType.DOC  # type: ignore[arg-type]
        except ImportError:
            _, extension = os.path.splitext(str(self.file_path))
            is_doc = extension == ".doc"

        if is_doc:
            validate_unstructured_version("0.4.11")

        if is_doc:
            from unstructured.partition.doc import partition_doc

            return partition_doc(filename=self.file_path, **self.unstructured_kwargs)  # type: ignore[arg-type]
        else:
            from unstructured.partition.docx import partition_docx

            return partition_docx(filename=self.file_path, **self.unstructured_kwargs)  # type: ignore[arg-type]
