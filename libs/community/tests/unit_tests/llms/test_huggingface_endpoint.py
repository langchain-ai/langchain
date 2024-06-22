"""Test `HuggingFaceEndpoint`."""

import pytest
from langchain_core.pydantic_v1 import ValidationError

from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint


def test_disable_api_token_check():
    with pytest.raises(ValidationError):
        HuggingFaceEndpoint(endpoint_url="some-url")

    HuggingFaceEndpoint(endpoint_url="some-url", validate_api_token=False)
