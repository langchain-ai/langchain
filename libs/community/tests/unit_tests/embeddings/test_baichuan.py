from typing import cast

from pydantic import SecretStr

from langchain_community.embeddings import BaichuanTextEmbeddings


def test_sparkllm_initialization_by_alias() -> None:
    # Effective initialization
    embeddings = BaichuanTextEmbeddings(
        model="embedding_model",
        api_key="your-api-key",  # type: ignore[arg-type]
        session=None,
    )
    assert embeddings.model_name == "embedding_model"
    assert (
        cast(SecretStr, embeddings.baichuan_api_key).get_secret_value()
        == "your-api-key"
    )
