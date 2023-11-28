import json
import urllib.request
from typing import List, Optional

from langchain_core.documents import Document

from langchain.document_loaders.base import BaseLoader
from langchain.utils import get_from_env, stringify_dict

STRIPE_ENDPOINTS = {
    "balance_transactions": "https://api.stripe.com/v1/balance_transactions",
    "charges": "https://api.stripe.com/v1/charges",
    "customers": "https://api.stripe.com/v1/customers",
    "events": "https://api.stripe.com/v1/events",
    "refunds": "https://api.stripe.com/v1/refunds",
    "disputes": "https://api.stripe.com/v1/disputes",
}


class StripeLoader(BaseLoader):
    """Load from `Stripe` API."""

    def __init__(self, resource: str, access_token: Optional[str] = None) -> None:
        """Initialize with a resource and an access token.

        Args:
            resource: The resource.
            access_token: The access token.
        """
        self.resource = resource
        access_token = access_token or get_from_env(
            "access_token", "STRIPE_ACCESS_TOKEN"
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
        endpoint = STRIPE_ENDPOINTS.get(self.resource)
        if endpoint is None:
            return []
        return self._make_request(endpoint)

    def load(self) -> List[Document]:
        return self._get_resource()
