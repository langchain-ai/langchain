from langchain_community.chat_loaders import _module_lookup

EXPECTED_ALL = [
    "BaseChatLoader",
    "FolderFacebookMessengerChatLoader",
    "GMailLoader",
    "IMessageChatLoader",
    "LangSmithDatasetChatLoader",
    "LangSmithRunChatLoader",
    "SingleFileFacebookMessengerChatLoader",
    "SlackChatLoader",
    "TelegramChatLoader",
    "WhatsAppChatLoader",
]


def test_all_imports() -> None:
    assert set(_module_lookup.keys()) == set(EXPECTED_ALL)
