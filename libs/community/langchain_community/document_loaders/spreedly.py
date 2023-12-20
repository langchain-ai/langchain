import json
import urllib.request
from typing import List

from langchain_core.documents import Document
from langchain_core.utils import stringify_dict

from langchain_community.document_loaders.base import BaseLoader

SPREEDLY_ENDPOINTS = {
    "gateways_options": "https://core.spreedly.com/v1/gateways_options.json",
    "gateways": "https://core.spreedly.com/v1/gateways.json",
    "receivers_options": "https://core.spreedly.com/v1/receivers_options.json",
    "receivers": "https://core.spreedly.com/v1/receivers.json",
    "payment_methods": "https://core.spreedly.com/v1/payment_methods.json",
    "certificates": "https://core.spreedly.com/v1/certificates.json",
    "transactions": "https://core.spreedly.com/v1/transactions.json",
    "environments": "https://core.spreedly.com/v1/environments.json",
}


class SpreedlyLoader(BaseLoader):
    """Load from `Spreedly` API."""

    def __init__(self, access_token: str, resource: str) -> None:
        """Initialize with an access token and a resource.

        Args:
            access_token: The access token.
            resource: The resource.
        """
        self.access_token = access_token
        self.resource = resource
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

    def _make_request(self, url: str) -> List[Document]:
        request = urllib.request.Request(url, headers=self.headers)

        with urllib.request.urlopen(request) as response:
            json_data = json.loads(response.read().decode())
            text = stringify_dict(json_data)
            metadata = {"source": url}
            return [Document(page_content=text, metadata=metadata)]

    def _get_resource(self) -> List[Document]:
        endpoint = SPREEDLY_ENDPOINTS.get(self.resource)
        if endpoint is None:
            return []
        return self._make_request(endpoint)

    def load(self) -> List[Document]:
        return self._get_resource()
