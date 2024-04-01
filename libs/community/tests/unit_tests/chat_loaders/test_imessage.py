import datetime
import pathlib

from langchain_community.chat_loaders import imessage, utils


def test_imessage_chat_loader_upgrade_osx11() -> None:
    chat_path = (
        pathlib.Path(__file__).parent / "data" / "imessage_chat_upgrade_osx_11.db"
    )
    loader = imessage.IMessageChatLoader(str(chat_path))

    chat_sessions = list(
        utils.map_ai_messages(loader.lazy_load(), sender="testemail@gmail.com")
    )
    assert chat_sessions, "Chat sessions should not be empty"

    assert chat_sessions[0]["messages"], "Chat messages should not be empty"

    first_message = chat_sessions[0]["messages"][0]
    # message content in text field
    assert "Yeh" in first_message.content, "Chat content mismatch"

    # time parsed correctly
    expected_message_time = 720845450393148160
    assert (
        first_message.additional_kwargs["message_time"] == expected_message_time
    ), "unexpected time"

    expected_parsed_time = datetime.datetime(2023, 11, 5, 2, 50, 50, 393148)
    assert (
        first_message.additional_kwargs["message_time_as_datetime"]
        == expected_parsed_time
    ), "date failed to parse"

    # is_from_me parsed correctly
    assert (
        first_message.additional_kwargs["is_from_me"] is False
    ), "is_from_me failed to parse"


def test_imessage_chat_loader() -> None:
    chat_path = pathlib.Path(__file__).parent / "data" / "imessage_chat.db"
    loader = imessage.IMessageChatLoader(str(chat_path))

    chat_sessions = list(
        utils.map_ai_messages(loader.lazy_load(), sender="testemail@gmail.com")
    )
    assert chat_sessions, "Chat sessions should not be empty"

    assert chat_sessions[0]["messages"], "Chat messages should not be empty"

    first_message = chat_sessions[0]["messages"][0]

    # message content in text field
    assert "Yeh" in first_message.content, "Chat content mismatch"

    # time parsed correctly
    expected_message_time = 720845450393148160
    assert (
        first_message.additional_kwargs["message_time"] == expected_message_time
    ), "unexpected time"

    expected_parsed_time = datetime.datetime(2023, 11, 5, 2, 50, 50, 393148)
    assert (
        first_message.additional_kwargs["message_time_as_datetime"]
        == expected_parsed_time
    ), "date failed to parse"

    # is_from_me parsed correctly
    assert (
        first_message.additional_kwargs["is_from_me"] is False
    ), "is_from_me failed to parse"

    # short message content in attributedBody field
    assert (
        "John is the almighty" in chat_sessions[0]["messages"][16].content
    ), "Chat content mismatch"

    # long message content in attributedBody field
    long_msg = "aaaaabbbbbaaaaabbbbbaaaaabbbbbaaaaabbbbbaaaaabbbbbaaaaabbbbbaaaaabbbbba"
    "aaaabbbbbaaaaabbbbbaaaaabbbbbaaaaabbbbbaaaaabbbbbaaaaabbb"
    assert long_msg in chat_sessions[0]["messages"][18].content, "Chat content mismatch"
