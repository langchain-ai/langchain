from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.chat_models.coze import ChatCoze

# For testing, run:
# TEST_FILE=tests/integration_tests/chat_models/test_coze.py make test


def test_chat_coze_default() -> None:
    chat = ChatCoze(
        coze_api_base="https://api.coze.com",
        coze_api_key="pat_...",  # type: ignore[arg-type]
        bot_id="7....",
        user="123",
        conversation_id="",
        streaming=True,
    )
    message = HumanMessage(content="请完整背诵将进酒，背诵5遍")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_chat_coze_default_non_streaming() -> None:
    chat = ChatCoze(
        coze_api_base="https://api.coze.com",
        coze_api_key="pat_...",  # type: ignore[arg-type]
        bot_id="7....",
        user="123",
        conversation_id="",
        streaming=False,
    )
    message = HumanMessage(content="请完整背诵将进酒，背诵5遍")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
