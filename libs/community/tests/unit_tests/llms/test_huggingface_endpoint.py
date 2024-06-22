"""Test `HuggingFaceEndpoint`."""

import pytest
from langchain_core.pydantic_v1 import ValidationError

from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint


@pytest.mark.requires("huggingface_hub")
def test_disable_api_token_check() -> None:
    with pytest.raises(ValidationError):
        HuggingFaceEndpoint(endpoint_url="some-url")  # type: ignore[call-arg]

    HuggingFaceEndpoint(endpoint_url="some-url", validate_api_token=False)  # type: ignore[call-arg]
