import os
from typing import Any, Dict, Iterator, List, Tuple, Union

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class OutlineLoader(BaseLoader):
    """Load `Outline` documents.

    This loader will use the Outline API to retrieve all documents in an Outline
    instance.  You will need the API key from Outline to configure the loader.
    The API used is documented here: https://www.getoutline.com/developers

    If not passed in as parameters the API key and will be taken from env
    vars OUTLINE_INSTANCE_URL and OUTLINE_API_KEY.

    Examples
    --------
    from langchain_community.document_loaders import OutlineLoader

    loader = OutlineLoader(
        outline_base_url="outlinewiki.somedomain.com", outline_api_key="theapikey"
    )
    docs = loader.load()
    """

    def __init__(
        self,
        outline_base_url: Union[str | None] = None,
        outline_api_key: Union[str | None] = None,
        page_size: int = 25,
    ):
        """Initialize with url, api_key and requested page size for API results
        pagination.

        :param outline_base_url: The URL of the outline instance.

        :param outline_api_key: API key for accessing the outline instance.

        :param page_size: How many outline documents should be retrieved per request
        """

        self.outline_base_url = outline_base_url or os.environ["OUTLINE_INSTANCE_URL"]
        self.outline_api_key = outline_api_key or os.environ["OUTLINE_API_KEY"]
        self.document_list_endpoint = f"{self.outline_base_url}/api/documents.list"
        self.page_size = page_size
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
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

    def _build_metadata(self, result: Any) -> Dict:
        metadata = {"source": f"{self.outline_base_url}/{result['url']}"}
        metadata["id"] = result["id"]
        metadata["title"] = result["title"]
        metadata["createdAt"] = result["createdAt"]
        metadata["updatedAt"] = result["updatedAt"]
        return metadata

    def _extract_pagination_info(self, pagination_data: Dict) -> Tuple[int, int]:
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

    def _fetch_all(self) -> Iterator[Dict]:
        starting_offset = 0

        offset, total_documents, page_entries = self._fetch_page(starting_offset)
        yield from page_entries

        while offset < total_documents:
            offset, _, page_entries = self._fetch_page(offset)
            yield from page_entries

    def _fetch_page(self, offset: int) -> Tuple[int, int, List[Dict]]:
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
