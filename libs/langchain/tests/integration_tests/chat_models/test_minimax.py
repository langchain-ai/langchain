from langchain.chat_models import MiniMaxChat


def test_minimaxchat_init_client():
    chat = MiniMaxChat(
        minimax_api_host="test_host",
        minimax_api_key="test_api_key",
        minimax_group_id="test_group_id",
    )
    assert chat._client
    assert chat._client.host == "test_host"
    assert chat._client.group_id == "test_group_id"
    assert chat._client.api_key == "test_api_key"
