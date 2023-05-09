"""Loader that loads local airbyte json files."""
import json
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utils import stringify_dict


class AirbyteJSONLoader(BaseLoader):
    """Loader that loads local airbyte json files."""

    def __init__(self, file_path: str):
        """Initialize with file path. This should start with '/tmp/airbyte_local/'."""
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load file."""
        text = ""
        for line in open(self.file_path, "r"):
            data = json.loads(line)["_airbyte_data"]
            text += stringify_dict(data)
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]
