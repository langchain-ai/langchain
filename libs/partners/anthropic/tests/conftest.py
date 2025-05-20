import pytest
from langchain_tests.conftest import _base_vcr_config as _base_vcr_config


def remove_response_headers(response: dict) -> dict:
    response["headers"] = {}
    return response


@pytest.fixture(scope="session")
def vcr_config(_base_vcr_config: dict) -> dict:  # noqa: F811
    """
    Extend the default configuration coming from langchain_tests.
    """
    config = _base_vcr_config.copy()
    config["before_record_response"] = remove_response_headers

    return config
