"""Test embedding model integration."""

import os
from typing import cast

from pydantic.v1 import SecretStr

from langchain_naver.embeddings import ClovaStudioEmbeddings

os.environ["NCP_CLOVASTUDIO_API_KEY"] = "test_api_key"
os.environ["NCP_APIGW_API_KEY"] = "test_gw_key"
os.environ["NCP_CLOVADTUDIO_APP_ID"] = "test_app_id"


def test_initialization_api_key() -> None:
    chat_model = ClovaStudioEmbeddings(clovastudio_api_key="foo", apigw_api_key="bar")
    assert cast(SecretStr, chat_model.ncp_clovastudio_api_key).get_secret_value() == "foo"
    assert cast(SecretStr, chat_model.ncp_apigw_api_key).get_secret_value() == "bar"