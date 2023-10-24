import pathlib

from langchain.chat_loaders import slack, utils


def test_slack_chat_loader() -> None:
    chat_path = (
        pathlib.Path(__file__).parents[2]
        / "integration_tests"
        / "examples"
        / "slack_export.zip"
    )
    loader = slack.SlackChatLoader(str(chat_path))

    chat_sessions = list(
        utils.map_ai_messages(loader.lazy_load(), sender="U0500003428")
    )
    assert chat_sessions, "Chat sessions should not be empty"

    assert chat_sessions[1]["messages"], "Chat messages should not be empty"

    assert (
        "Example message" in chat_sessions[1]["messages"][0].content
    ), "Chat content mismatch"
