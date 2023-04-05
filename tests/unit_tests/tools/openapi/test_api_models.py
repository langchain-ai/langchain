"""Test the APIOperation class."""


import pytest

from langchain.tools.openapi.utils.api_models import APIOperation
from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec


_ROBOT_METHODS = [
    ("/ask_for_help", "post"),
    ("/ask_for_passphrase", "get"),
    ("/get_state", "get"),
    ("/goto/{x}/{y}/{z}", "post"),  # Private type definitions required.
    ("/recycle", "delete"),
    ("/walk", "post"),
]


@pytest.mark.parametrize("path, method", _ROBOT_METHODS)
def test_parse_api_operations(robot_spec: OpenAPISpec, path: str, method: str) -> None:
    """Test the APIOperation class."""
    APIOperation.from_openapi_spec(robot_spec, path, method)
