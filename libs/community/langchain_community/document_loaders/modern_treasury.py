import json
import urllib.request
from base64 import b64encode
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.utils import get_from_env, stringify_value

from langchain_community.document_loaders.base import BaseLoader

MODERN_TREASURY_ENDPOINTS = {
    "payment_orders": "https://app.moderntreasury.com/api/payment_orders",
    "expected_payments": "https://app.moderntreasury.com/api/expected_payments",
    "returns": "https://app.moderntreasury.com/api/returns",
    "incoming_payment_details": "https://app.moderntreasury.com/api/\
incoming_payment_details",
    "counterparties": "https://app.moderntreasury.com/api/counterparties",
    "internal_accounts": "https://app.moderntreasury.com/api/internal_accounts",
    "external_accounts": "https://app.moderntreasury.com/api/external_accounts",
    "transactions": "https://app.moderntreasury.com/api/transactions",
    "ledgers": "https://app.moderntreasury.com/api/ledgers",
    "ledger_accounts": "https://app.moderntreasury.com/api/ledger_accounts",
    "ledger_transactions": "https://app.moderntreasury.com/api/ledger_transactions",
    "events": "https://app.moderntreasury.com/api/events",
    "invoices": "https://app.moderntreasury.com/api/invoices",
}


class ModernTreasuryLoader(BaseLoader):
    """Load from `Modern Treasury`."""

    def __init__(
        self,
        resource: str,
        organization_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """

        Args:
            resource: The Modern Treasury resource to load.
            organization_id: The Modern Treasury organization ID. It can also be
               specified via the environment variable
               "MODERN_TREASURY_ORGANIZATION_ID".
            api_key: The Modern Treasury API key. It can also be specified via
               the environment variable "MODERN_TREASURY_API_KEY".
        """
        self.resource = resource
        organization_id = organization_id or get_from_env(
            "organization_id", "MODERN_TREASURY_ORGANIZATION_ID"
        )
        api_key = api_key or get_from_env("api_key", "MODERN_TREASURY_API_KEY")
        credentials = f"{organization_id}:{api_key}".encode("utf-8")
        basic_auth_token = b64encode(credentials).decode("utf-8")
        self.headers = {"Authorization": f"Basic {basic_auth_token}"}

    def _make_request(self, url: str) -> List[Document]:
        request = urllib.request.Request(url, headers=self.headers)

        with urllib.request.urlopen(request) as response:
            json_data = json.loads(response.read().decode())
            text = stringify_value(json_data)
            metadata = {"source": url}
            return [Document(page_content=text, metadata=metadata)]

    def _get_resource(self) -> List[Document]:
        endpoint = MODERN_TREASURY_ENDPOINTS.get(self.resource)
        if endpoint is None:
            return []
        return self._make_request(endpoint)

    def load(self) -> List[Document]:
        return self._get_resource()
