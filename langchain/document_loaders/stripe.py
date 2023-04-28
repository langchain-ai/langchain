"""Loader that fetches data from Stripe"""
import json
import urllib.request
from typing import Any, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

STRIPE_ENDPOINTS = {
    "balance_transactions": "https://api.stripe.com/v1/balance_transactions",
    "charges": "https://api.stripe.com/v1/charges",
    "customers": "https://api.stripe.com/v1/customers",
    "events": "https://api.stripe.com/v1/events",
    "refunds": "https://api.stripe.com/v1/refunds",
    "disputes": "https://api.stripe.com/v1/disputes",
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


class StripeLoader(BaseLoader):
    def __init__(self, access_token: str, resource: str) -> None:
        self.access_token = access_token
        self.resource = resource
        self.headers = {"Authorization": f"Bearer {self.access_token}"}

    def _make_request(self, url: str) -> List[Document]:
        request = urllib.request.Request(url, headers=self.headers)

        with urllib.request.urlopen(request) as response:
            json_data = json.loads(response.read().decode())
            text = _stringify_dict(json_data)
            metadata = {"source": url}
            return [Document(page_content=text, metadata=metadata)]

    def _get_resource(self) -> List[Document]:
        endpoint = STRIPE_ENDPOINTS.get(self.resource)
        if endpoint is None:
            return []
        return self._make_request(endpoint)

    def load(self) -> List[Document]:
        return self._get_resource()
