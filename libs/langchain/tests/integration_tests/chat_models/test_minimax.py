from langchain.chat_models import MiniMaxChat
from langchain.llms.minimax import MinimaxCommon


def test_minimaxchat_init_client() -> None:
    "Test MiniMaxChat and MinimaxCommon model init _client attribute"
    chat = MiniMaxChat(
        minimax_api_host="test_host",
        minimax_api_key="test_api_key",
        minimax_group_id="test_group_id",
    )
    assert chat._client
    assert chat._client.host == "test_host"
    assert chat._client.group_id == "test_group_id"
    assert chat._client.api_key == "test_api_key"

    chat2 = MinimaxCommon(
        minimax_api_host="test_host",
        minimax_api_key="test_api_key",
        minimax_group_id="test_group_id",
    )
    assert chat2._client
    assert chat2._client.host == "test_host"
    assert chat2._client.group_id == "test_group_id"
    assert chat2._client.api_key == "test_api_key"
