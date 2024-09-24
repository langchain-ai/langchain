"""Test embedding model integration."""

import os
from typing import cast

from langchain_core.pydantic_v1 import SecretStr

from langchain_community.embeddings import ClovaXEmbeddings

os.environ["NCP_CLOVASTUDIO_API_KEY"] = "test_api_key"
os.environ["NCP_APIGW_API_KEY"] = "test_gw_key"
os.environ["NCP_CLOVASTUDIO_APP_ID"] = "test_app_id"


def test_initialization_api_key() -> None:
    llm = ClovaXEmbeddings(api_key="foo", apigw_api_key="bar")  # type: ignore[arg-type]
    assert cast(SecretStr, llm.ncp_clovastudio_api_key).get_secret_value() == "foo"
    assert cast(SecretStr, llm.ncp_apigw_api_key).get_secret_value() == "bar"
