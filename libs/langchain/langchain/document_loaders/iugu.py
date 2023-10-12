import json
import urllib.request
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.utils import get_from_env, stringify_dict

IUGU_ENDPOINTS = {
    "invoices": "https://api.iugu.com/v1/invoices",
    "customers": "https://api.iugu.com/v1/customers",
    "charges": "https://api.iugu.com/v1/charges",
    "subscriptions": "https://api.iugu.com/v1/subscriptions",
    "plans": "https://api.iugu.com/v1/plans",
}


class IuguLoader(BaseLoader):
    """Load from `IUGU`."""

    def __init__(self, resource: str, api_token: Optional[str] = None) -> None:
        """Initialize the IUGU resource.

        Args:
            resource: The name of the resource to fetch.
            api_token: The IUGU API token to use.
        """
        self.resource = resource
        api_token = api_token or get_from_env("api_token", "IUGU_API_TOKEN")
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def _make_request(self, url: str) -> List[Document]:
        request = urllib.request.Request(url, headers=self.headers)

        with urllib.request.urlopen(request) as response:
            json_data = json.loads(response.read().decode())
            text = stringify_dict(json_data)
            metadata = {"source": url}
            return [Document(page_content=text, metadata=metadata)]

    def _get_resource(self) -> List[Document]:
        endpoint = IUGU_ENDPOINTS.get(self.resource)
        if endpoint is None:
            return []
        return self._make_request(endpoint)

    def load(self) -> List[Document]:
        return self._get_resource()
