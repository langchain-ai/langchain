"""Loader that loads Figma files json dump."""
import json
import urllib.request
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


class FigmaFileLoader(BaseLoader):
    """Loader that loads Figma file json."""

    def __init__(self, access_token: str, ids: str, key: str):
        """Initialize with access token, ids, and key."""
        self.access_token = access_token
        self.ids = ids
        self.key = key

    def _construct_figma_api_url(self) -> str:
        api_url = "https://api.figma.com/v1/files/%s/nodes?ids=%s" % (
            self.key,
            self.ids,
        )
        return api_url

    def _get_figma_file(self) -> Any:
        """Get Figma file from Figma REST API."""
        headers = {"X-Figma-Token": self.access_token}
        request = urllib.request.Request(
            self._construct_figma_api_url(), headers=headers
        )
        with urllib.request.urlopen(request) as response:
            json_data = json.loads(response.read().decode())
            return json_data

    def load(self) -> List[Document]:
        """Load file"""
        data = self._get_figma_file()
        text = _stringify_dict(data)
        metadata = {"source": self._construct_figma_api_url()}
        return [Document(page_content=text, metadata=metadata)]
