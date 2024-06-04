import os
from typing import cast

from langchain_core.pydantic_v1 import SecretStr

from langchain_community.embeddings import MiniMaxEmbeddings


def test_minimax_non_group_id() -> None:
    """Test no longer using group_id"""
    # Setting environment variable
    os.environ["MINIMAX_API_KEY"] = "your-api-key"

    # If exist, delete the "MINIMAX_GROUP_ID"
    if "MINIMAX_GROUP_ID" in os.environ:
        del os.environ["MINIMAX_GROUP_ID"]

    # Effective initialization
    MiniMaxEmbeddings()


def test_minimax_standard_init_args() -> None:
    """Test standardized model init arguments"""
    # Effective initialization
    embeddings = MiniMaxEmbeddings(
        api_key="your-api-key"  # type: ignore[arg-type]
    )
    assert (
        cast(SecretStr, embeddings.minimax_api_key).get_secret_value() == "your-api-key"
    )
