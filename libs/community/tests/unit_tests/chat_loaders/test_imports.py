from langchain_community.chat_loaders import __all__, _module_lookup

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
    """Test that __all__ is correctly set."""
    assert set(__all__) == set(EXPECTED_ALL)
    assert set(__all__) == set(_module_lookup.keys())
