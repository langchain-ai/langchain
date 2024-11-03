import re
from typing import Dict, Iterator, List

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class YuqueLoader(BaseLoader):
    """Load documents from `Yuque`."""

    def __init__(self, access_token: str, api_url: str = "https://www.yuque.com"):
        """Initialize with Yuque access_token and api_url.

        Args:
            access_token: Personal access token - see https://www.yuque.com/settings/tokens.
            api_url: Yuque API url.
        """
        self.access_token = access_token
        self.api_url = api_url

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Auth-Token": self.access_token,
        }

    def get_user_id(self) -> int:
        url = f"{self.api_url}/api/v2/user"
        response = self.http_get(url=url)

        return response["data"]["id"]

    def get_books(self, user_id: int) -> List[Dict]:
        url = f"{self.api_url}/api/v2/users/{user_id}/repos"
        response = self.http_get(url=url)

        return response["data"]

    def get_document_ids(self, book_id: int) -> List[int]:
        url = f"{self.api_url}/api/v2/repos/{book_id}/docs"
        response = self.http_get(url=url)

        return [document["id"] for document in response["data"]]

    def get_document(self, book_id: int, document_id: int) -> Dict:
        url = f"{self.api_url}/api/v2/repos/{book_id}/docs/{document_id}"
        response = self.http_get(url=url)

        return response["data"]

    def parse_document(self, document: Dict) -> Document:
        content = self.parse_document_body(document["body"])
        metadata = {
            "title": document["title"],
            "description": document["description"],
            "created_at": document["created_at"],
            "updated_at": document["updated_at"],
        }

        return Document(page_content=content, metadata=metadata)

    @staticmethod
    def parse_document_body(body: str) -> str:
        result = re.sub(r'<a name="(.*)"></a>', "", body)
        result = re.sub(r"<br\s*/?>", "", result)

        return result

    def http_get(self, url: str) -> Dict:
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()

    def get_documents(self) -> Iterator[Document]:
        user_id = self.get_user_id()
        books = self.get_books(user_id)

        for book in books:
            book_id = book["id"]
            document_ids = self.get_document_ids(book_id)
            for document_id in document_ids:
                document = self.get_document(book_id, document_id)
                parsed_document = self.parse_document(document)
                yield parsed_document

    def load(self) -> List[Document]:
        """Load documents from `Yuque`."""
        return list(self.get_documents())
