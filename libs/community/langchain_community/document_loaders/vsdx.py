import os
import tempfile
from abc import ABC
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

import requests

from langchain_community.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers import VsdxParser


class VsdxLoader(BaseLoader, ABC):
    def __init__(self, file_path: Union[str, Path]):
        """Initialize with file path."""
        self.file_path = str(file_path)
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

        self.parser = VsdxParser()  # type: ignore[misc]

    def __del__(self) -> None:
        if hasattr(self, "temp_file"):
            self.temp_file.close()

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if the url is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    def load(self) -> List[Document]:
        blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        return list(self.parser.parse(blob))
