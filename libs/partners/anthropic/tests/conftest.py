from typing import Any

import pytest
from langchain_tests.conftest import YamlGzipSerializer
from langchain_tests.conftest import _base_vcr_config as _base_vcr_config
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
def vcr_config(_base_vcr_config: dict) -> dict:  # noqa: F811
    """
    Extend the default configuration coming from langchain_tests.
    """
    config = _base_vcr_config.copy()
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"
    config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")

    return config


@pytest.fixture
def vcr(vcr_config: dict) -> VCR:
    """Override the default vcr fixture to include custom serializers"""
    my_vcr = VCR(**vcr_config)
    my_vcr.register_serializer("yaml.gz", YamlGzipSerializer)
    return my_vcr
