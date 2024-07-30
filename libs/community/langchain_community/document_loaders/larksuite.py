import json
import urllib.request
from typing import Any, Iterator

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class LarkSuiteDocLoader(BaseLoader):
    """Load from `LarkSuite` (`FeiShu`)."""

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


class LarkSuiteWikiLoader(LarkSuiteDocLoader):
    """Load from `LarkSuite` (`FeiShu`) wiki."""

    def __init__(self, domain: str, access_token: str, wiki_id: str):
        """Initialize with domain, access_token (tenant / user), and wiki_id.

        Args:
            domain: The domain to load the LarkSuite.
            access_token: The access_token to use.
            wiki_id: The wiki_id to load.
        """
        self.domain = domain
        self.access_token = access_token
        self.wiki_id = wiki_id
        self.document_id = ""

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load LarkSuite (FeiShu) wiki document."""

        # convert Feishu wiki id to document id
        if not self.document_id:
            wiki_url_prefix = f"{self.domain}/open-apis/wiki/v2/spaces/get_node"
            wiki_node_info_json = self._get_larksuite_api_json_data(
                f"{wiki_url_prefix}?token={self.wiki_id}"
            )
            self.document_id = wiki_node_info_json["data"]["node"]["obj_token"]

        yield from super().lazy_load()
