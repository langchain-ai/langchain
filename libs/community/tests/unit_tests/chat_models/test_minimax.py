import os

from langchain_community.chat_models import MiniMaxChat


def test_minimax_non_group_id() -> None:
    """Test no longer using group_id"""
    # Setting environment variable
    os.environ["MINIMAX_API_KEY"] = "your-api-key"

    # If exist, delete the "MINIMAX_GROUP_ID"
    if "MINIMAX_GROUP_ID" in os.environ:
        del os.environ["MINIMAX_GROUP_ID"]

    # Effective initialization
    MiniMaxChat()
