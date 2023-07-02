"""Loads LarkSuite (FeiShu) document json dump."""
import json
import urllib.request
from typing import Any, Iterator, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class LarkSuiteDocLoader(BaseLoader):
    """Loads LarkSuite (FeiShu) document."""

    def __init__(self, domain: str, access_token: str, document_id: str):
        """Initialize with domain, access_token (tenant / user), and document_id.

        Args:
            domain: The domain to load the LarkSuite.
            access_token: The access_token to use.
            document_id: The document_id to load.
        """
        self.domain = domain
        self.access_token = access_token
        self.document_id = document_id

    def _get_larksuite_api_json_data(self, api_url: str) -> Any:
        """Get LarkSuite (FeiShu) API response json data."""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        request = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(request) as response:
            json_data = json.loads(response.read().decode())
            return json_data

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load LarkSuite (FeiShu) document."""
        api_url_prefix = f"{self.domain}/open-apis/docx/v1/documents"
        metadata_json = self._get_larksuite_api_json_data(
            f"{api_url_prefix}/{self.document_id}"
        )
        raw_content_json = self._get_larksuite_api_json_data(
            f"{api_url_prefix}/{self.document_id}/raw_content"
        )
        text = raw_content_json["data"]["content"]
        metadata = {
            "document_id": self.document_id,
            "revision_id": metadata_json["data"]["document"]["revision_id"],
            "title": metadata_json["data"]["document"]["title"],
        }
        yield Document(page_content=text, metadata=metadata)

    def load(self) -> List[Document]:
        """Load LarkSuite (FeiShu) document."""
        return list(self.lazy_load())
