"""Loader that fetches data from Spotify"""
import json
import urllib.request
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utils import get_from_env, stringify_dict

SPOTIFY_ENDPOINTS = {
    "tracks": "https://api.spotify.com/v1/me/tracks",
}

class SpotifyLoader(BaseLoader):
    def __init__(self, resource: str, access_token: Optional[str] = None) -> None:
        self.resource = resource
        access_token = access_token or get_from_env(
            "access_token", "SPOTIFY_ACCESS_TOKEN"
        )
        self.headers = {"Authorization": f"Bearer {access_token}"}

    def _make_request(self, url: str) -> List[Document]:
        request = urllib.request.Request(url, headers=self.headers)

        with urllib.request.urlopen(request) as response:
            json_data = json.loads(response.read().decode())
            text = stringify_dict(json_data)
            metadata = {"source": url}
            return [Document(page_content=text, metadata=metadata)]

    def _get_resource(self) -> List[Document]:
        endpoint = SPOTIFY_ENDPOINTS.get(self.resource)
        if endpoint is None:
            return []
        return self._make_request(endpoint)

    def load(self) -> List[Document]:
        return self._get_resource()
