import pathlib

from langchain.chat_loaders import imessage, utils


def test_imessage_chat_loader() -> None:
    chat_path = pathlib.Path(__file__).parent / "data" / "imessage_chat.db"
    loader = imessage.IMessageChatLoader(str(chat_path))

    chat_sessions = list(
        utils.map_ai_messages(loader.lazy_load(), sender="testemail@gmail.com")
    )
    assert chat_sessions, "Chat sessions should not be empty"

    assert chat_sessions[0]["messages"], "Chat messages should not be empty"

    # message content in text field
    assert "Yeh" in chat_sessions[0]["messages"][0].content, "Chat content mismatch"

    # short message content in attributedBody field
    assert (
        "John is the almighty" in chat_sessions[0]["messages"][16].content
    ), "Chat content mismatch"

    # long message content in attributedBody field
    long_msg = "aaaaabbbbbaaaaabbbbbaaaaabbbbbaaaaabbbbbaaaaabbbbbaaaaabbbbbaaaaabbbbba"
    "aaaabbbbbaaaaabbbbbaaaaabbbbbaaaaabbbbbaaaaabbbbbaaaaabbb"
    assert long_msg in chat_sessions[0]["messages"][18].content, "Chat content mismatch"
