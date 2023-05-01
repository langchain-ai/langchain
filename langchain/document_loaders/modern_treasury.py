"""Loader that fetches data from Modern Treasury"""
import json
import urllib.request
from base64 import b64encode
from typing import Any, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

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


def _stringify_value(val: Any) -> str:
    if isinstance(val, str):
        return val
    elif isinstance(val, dict):
        return "\n" + _stringify_dict(val)
    elif isinstance(val, list):
        return "\n".join(_stringify_value(v) for v in val)
    else:
        return str(val)


def _stringify_dict(data: dict) -> str:
    text = ""
    for key, value in data.items():
        text += key + ": " + _stringify_value(value) + "\n"
    return text


class ModernTreasuryLoader(BaseLoader):
    def __init__(self, organization_id: str, api_key: str, resource: str) -> None:
        self.organization_id = organization_id
        self.api_key = api_key
        self.resource = resource

        credentials = f"{self.organization_id}:{self.api_key}".encode('utf-8')
        self.basic_auth_token = b64encode(credentials).decode('utf-8')
        self.headers = {"Authorization": f"Basic {self.basic_auth_token}"}

    def _make_request(self, url: str) -> List[Document]:
        request = urllib.request.Request(url, headers=self.headers)

        with urllib.request.urlopen(request) as response:
            json_data = json.loads(response.read().decode())
            text = _stringify_value(json_data)
            metadata = {"source": url}
            return [Document(page_content=text, metadata=metadata)]

    def _get_resource(self) -> List[Document]:
        endpoint = MODERN_TREASURY_ENDPOINTS.get(self.resource)
        if endpoint is None:
            return []
        return self._make_request(endpoint)

    def load(self) -> List[Document]:
        return self._get_resource()
