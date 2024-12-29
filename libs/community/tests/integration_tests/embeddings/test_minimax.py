from typing import cast

from pydantic import SecretStr

from langchain_community.embeddings import MiniMaxEmbeddings


def test_initialization_with_alias() -> None:
    """Test minimax embedding model initialization with alias."""
    api_key = "your-api-key"
    group_id = "your-group-id"

    embeddings = MiniMaxEmbeddings(  # type: ignore[arg-type, call-arg]
        api_key=api_key,  # type: ignore[arg-type]
        group_id=group_id,  # type: ignore[arg-type]
    )

    assert cast(SecretStr, embeddings.minimax_api_key).get_secret_value() == api_key
    assert embeddings.minimax_group_id == group_id
