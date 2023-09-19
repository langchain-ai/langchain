from pathlib import Path

import pytest

from langchain.document_loaders import TelegramChatApiLoader, TelegramChatFileLoader


def test_telegram_chat_file_loader() -> None:
    """Test TelegramChatFileLoader."""
    file_path = Path(__file__).parent / "test_docs" / "telegram.json"
    loader = TelegramChatFileLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["source"] == str(file_path)
    assert docs[0].page_content == (
        "Henry on 2020-01-01T00:00:02: It's 2020...\n\n"
        "Henry on 2020-01-01T00:00:04: Fireworks!\n\n"
        "Grace ðŸ§¤ ðŸ\x8d’ on 2020-01-01T00:00:05: You're a minute late!\n\n"
    )


@pytest.mark.requires("telethon", "pandas")
def test_telegram_channel_loader_parsing() -> None:
    """Test TelegramChatApiLoader."""
    file_path = Path(__file__).parent / "test_docs" / "telegram_channel.json"
    # if we don't provide any value, it will skip fetching from telegram
    # and will check the parsing logic.
    loader = TelegramChatApiLoader(file_path=str(file_path))
    docs = loader.load()

    assert len(docs) == 1
    print(docs[0].page_content)
    assert docs[0].page_content == (
        "Hello, world!.\nLLMs are awesome! Langchain is great. Telegram is the best!."
    )
