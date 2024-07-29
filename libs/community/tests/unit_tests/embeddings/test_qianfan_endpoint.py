from typing import cast

from langchain_core.pydantic_v1 import SecretStr

from langchain_community.embeddings import QianfanEmbeddingsEndpoint


def test_initialization_with_alias() -> None:
    """Test qianfan embedding model initialization with alias."""
    api_key = "your-api-key"
    secret_key = "your-secret-key"

    embeddings = QianfanEmbeddingsEndpoint(  # type: ignore[arg-type, call-arg]
        api_key=api_key, secret_key=secret_key
    )

    assert cast(SecretStr, embeddings.qianfan_ak).get_secret_value() == api_key
    assert cast(SecretStr, embeddings.qianfan_sk).get_secret_value() == secret_key
