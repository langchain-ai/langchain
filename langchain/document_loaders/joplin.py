import json
import urllib
from datetime import datetime
from typing import List, Optional

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from langchain.utils import get_from_env


class JoplinLoader(BaseLoader):
    """
    Loader that fetches notes from Joplin.

    In order to use this loader, you need to have Joplin running with the
    Web Clipper enabled (look for "Web Clipper" in the app settings).

    To get the access token, you need to go to the Web Clipper options and
    under "Advanced Options" you will find the access token.

    You can find more information about the Web Clipper service here:
    https://joplinapp.org/clipper/
    """

    def __init__(self, access_token: Optional[str] = None) -> None:
        self.access_token = access_token or get_from_env(
            "access_token", "JOPLIN_ACCESS_TOKEN"
        )
        self.get_note = (
            f"http://localhost:41184/notes?token={self.access_token}&fields=id,parent_id,title,body,created_time,updated_time&page="
            + "{page}"
        )
        self.get_folder = (
            "http://localhost:41184/folders/{id}"
            + f"?token={self.access_token}&fields=title"
        )
        self.get_tag = (
            "http://localhost:41184/notes/{id}/tags"
            + f"?token={self.access_token}&fields=title"
        )
        self.link_note = "joplin://x-callback-url/openNote?id={id}"

    def _get_notes(self) -> List[Document]:
        has_more = True
        page = 1

        notes = []
        while has_more:
            req_note = urllib.request.Request(self.get_note.format(page=page))
            with urllib.request.urlopen(req_note) as response:
                json_data = json.loads(response.read().decode())
                for note in json_data["items"]:
                    metadata = {
                        "source": self.link_note.format(id=note["id"]),
                        "folder": self._get_folder(note["parent_id"]),
                        "tags": self._get_tags(note["id"]),
                        "title": note["title"],
                        "created_time": self._convert_date(note["created_time"]),
                        "updated_time": self._convert_date(note["updated_time"]),
                    }
                    notes.append(Document(page_content=note["body"], metadata=metadata))

                has_more = json_data["has_more"]
                page += 1

        return notes

    def _get_folder(self, folder_id: str) -> str:
        req_folder = urllib.request.Request(self.get_folder.format(id=folder_id))
        with urllib.request.urlopen(req_folder) as response:
            json_data = json.loads(response.read().decode())
            return json_data["title"]

    def _get_tags(self, note_id: str) -> List[str]:
        req_tag = urllib.request.Request(self.get_tag.format(id=note_id))
        with urllib.request.urlopen(req_tag) as response:
            json_data = json.loads(response.read().decode())
            return [tag["title"] for tag in json_data["items"]]

    def _convert_date(self, date: int) -> str:
        return datetime.fromtimestamp(date / 1000).strftime("%Y-%m-%d %H:%M:%S")

    def load(self) -> List[Document]:
        return self._get_notes()
