from typing import Any, Iterator, List

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class OutlineLoader(BaseLoader):
    def __init__(
        self, outline_base_url: str, outline_api_key: str, page_size: int = 25
    ):
        self.outline_base_url = outline_base_url
        self.outline_api_key = outline_api_key
        self.document_list_endpoint = f"{self.outline_base_url}/api/documents.list"
        self.page_size = page_size
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {outline_api_key}",
        }

    def lazy_load(self) -> Iterator[Document]:
        """
        Loads documents from Outline.
        """
        results = self._fetch_all()
        for result in results:
            text = result["text"]
            metadata = self._build_metadata(result)
            yield Document(page_content=text, metadata=metadata)

    def _build_metadata(self, result: Any) -> dict:
        metadata = {"source": f"{self.outline_base_url}/{result['url']}"}
        metadata["id"] = result["id"]
        metadata["title"] = result["title"]
        metadata["createdAt"] = result["createdAt"]
        metadata["updatedAt"] = result["updatedAt"]
        return metadata

    def _extract_pagination_info(self, pagination_data: dict) -> tuple[int, int]:
        next_path = pagination_data.get("nextPath", "")
        next_offset = 0
        if next_path:
            try:
                offset_str = next_path.split("offset=")[1].split("&")[0]
                next_offset = int(offset_str)
            except (IndexError, ValueError):
                next_offset = 0

        total = pagination_data.get("total", 0)

        return next_offset, total

    def _fetch_all(self) -> Iterator[dict]:
        starting_offset = 0
        offset, total_documents, data = self._fetch_page(starting_offset)
        yield from data
        while offset < total_documents:
            offset, _, data = self._fetch_page(offset)
            yield from data

    def _fetch_page(self, offset: int) -> tuple[int, int, List[dict]]:
        payload = {
            "offset": offset,
            "limit": self.page_size,
            "sort": "updatedAt",
            "direction": "DESC",
            "query": "",
        }
        response = requests.post(
            self.document_list_endpoint, json=payload, headers=self.headers
        )
        response.raise_for_status()
        response_json = response.json()
        offset, total_documents = self._extract_pagination_info(
            response_json["pagination"]
        )
        return offset, total_documents, response_json["data"]
