"""Loader that loads local airbyte json files."""
import json
from typing import Any, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


def _stringify_value(val: Any) -> str:
    if isinstance(val, str):
        return val
    elif isinstance(val, dict):
        return "\n" + _stringify_dict(val)
    elif isinstance(val, list):
        return "\n".join(_stringify_value(v) for v in val)
    else:
        return str(val)


def _stringify_dict(data: dict) -> str:
    text = ""
    for key, value in data.items():
        text += key + ": " + _stringify_value(data[key]) + "\n"
    return text


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
            text += _stringify_dict(data)
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]
