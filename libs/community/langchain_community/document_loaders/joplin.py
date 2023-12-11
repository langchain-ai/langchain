import json
import urllib
from datetime import datetime
from typing import Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.utils import get_from_env

from langchain_community.document_loaders.base import BaseLoader

LINK_NOTE_TEMPLATE = "joplin://x-callback-url/openNote?id={id}"


class JoplinLoader(BaseLoader):
    """Load notes from `Joplin`.

    In order to use this loader, you need to have Joplin running with the
    Web Clipper enabled (look for "Web Clipper" in the app settings).

    To get the access token, you need to go to the Web Clipper options and
    under "Advanced Options" you will find the access token.

    You can find more information about the Web Clipper service here:
    https://joplinapp.org/clipper/
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        port: int = 41184,
        host: str = "localhost",
    ) -> None:
        """

        Args:
            access_token: The access token to use.
            port: The port where the Web Clipper service is running. Default is 41184.
            host: The host where the Web Clipper service is running.
                Default is localhost.
        """
        access_token = access_token or get_from_env(
            "access_token", "JOPLIN_ACCESS_TOKEN"
        )
        base_url = f"http://{host}:{port}"
        self._get_note_url = (
            f"{base_url}/notes?token={access_token}"
            f"&fields=id,parent_id,title,body,created_time,updated_time&page={{page}}"
        )
        self._get_folder_url = (
            f"{base_url}/folders/{{id}}?token={access_token}&fields=title"
        )
        self._get_tag_url = (
            f"{base_url}/notes/{{id}}/tags?token={access_token}&fields=title"
        )

    def _get_notes(self) -> Iterator[Document]:
        has_more = True
        page = 1
        while has_more:
            req_note = urllib.request.Request(self._get_note_url.format(page=page))
            with urllib.request.urlopen(req_note) as response:
                json_data = json.loads(response.read().decode())
                for note in json_data["items"]:
                    metadata = {
                        "source": LINK_NOTE_TEMPLATE.format(id=note["id"]),
                        "folder": self._get_folder(note["parent_id"]),
                        "tags": self._get_tags(note["id"]),
                        "title": note["title"],
                        "created_time": self._convert_date(note["created_time"]),
                        "updated_time": self._convert_date(note["updated_time"]),
                    }
                    yield Document(page_content=note["body"], metadata=metadata)

                has_more = json_data["has_more"]
                page += 1

    def _get_folder(self, folder_id: str) -> str:
        req_folder = urllib.request.Request(self._get_folder_url.format(id=folder_id))
        with urllib.request.urlopen(req_folder) as response:
            json_data = json.loads(response.read().decode())
            return json_data["title"]

    def _get_tags(self, note_id: str) -> List[str]:
        req_tag = urllib.request.Request(self._get_tag_url.format(id=note_id))
        with urllib.request.urlopen(req_tag) as response:
            json_data = json.loads(response.read().decode())
            return [tag["title"] for tag in json_data["items"]]

    def _convert_date(self, date: int) -> str:
        return datetime.fromtimestamp(date / 1000).strftime("%Y-%m-%d %H:%M:%S")

    def lazy_load(self) -> Iterator[Document]:
        yield from self._get_notes()

    def load(self) -> List[Document]:
        return list(self.lazy_load())
