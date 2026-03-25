import json
from typing import Any

import pytest
from langchain_tests.conftest import CustomPersister, CustomSerializer, base_vcr_config
from vcr import VCR  # type: ignore[import-untyped]

_EXTRA_HEADERS = [
    ("openai-organization", "PLACEHOLDER"),
    ("user-agent", "PLACEHOLDER"),
    ("x-openai-client-user-agent", "PLACEHOLDER"),
]


def remove_request_headers(request: Any) -> Any:
    """Remove sensitive headers from the request."""
    for k in request.headers:
        request.headers[k] = "**REDACTED**"
    request.uri = "**REDACTED**"
    return request


def remove_response_headers(response: dict) -> dict:
    """Remove sensitive headers from the response."""
    for k in response["headers"]:
        response["headers"][k] = "**REDACTED**"
    return response


@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Extend the default configuration coming from langchain_tests."""
    config = base_vcr_config()
    config["match_on"] = [
        m if m != "body" else "json_body" for m in config.get("match_on", [])
    ]
    config.setdefault("filter_headers", []).extend(_EXTRA_HEADERS)
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"
    config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")
    return config


def _json_body_matcher(r1: Any, r2: Any) -> None:
    """Match request bodies as parsed JSON, ignoring key order."""
    b1 = r1.body or b""
    b2 = r2.body or b""
    if isinstance(b1, bytes):
        b1 = b1.decode("utf-8")
    if isinstance(b2, bytes):
        b2 = b2.decode("utf-8")
    try:
        j1 = json.loads(b1)
        j2 = json.loads(b2)
    except (json.JSONDecodeError, ValueError):
        assert b1 == b2, f"body mismatch (non-JSON):\n{b1}\n!=\n{b2}"
        return
    assert j1 == j2, f"body mismatch:\n{j1}\n!=\n{j2}"


def pytest_recording_configure(config: dict, vcr: VCR) -> None:
    vcr.register_persister(CustomPersister())
    vcr.register_serializer("yaml.gz", CustomSerializer())
    vcr.register_matcher("json_body", _json_body_matcher)
