from typing import Any

import pytest
from langchain_tests.conftest import CustomPersister, CustomSerializer, base_vcr_config
from vcr import VCR  # type: ignore[import-untyped]


def remove_request_headers(request: Any) -> Any:
    for k in request.headers:
        request.headers[k] = "**REDACTED**"
    return request


def remove_response_headers(response: dict) -> dict:
    for k in response["headers"]:
        response["headers"][k] = "**REDACTED**"
    return response


@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Extend the default configuration coming from langchain_tests."""
    config = base_vcr_config()
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"
    config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")

    return config


def pytest_recording_configure(config: dict, vcr: VCR) -> None:
    vcr.register_persister(CustomPersister())
    vcr.register_serializer("yaml.gz", CustomSerializer())
