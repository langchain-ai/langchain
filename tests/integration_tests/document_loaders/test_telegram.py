from pathlib import Path

from langchain.document_loaders import TelegramChatLoader


def test_telegram_chat_loader() -> None:
    """Test TelegramChatLoader."""
    file_path = Path(__file__).parent.parent / "examples/telegram.json"
    loader = TelegramChatLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["source"] == str(file_path)
    assert docs[0].page_content == (
        "Henry on 2020-01-01T00:00:02: It's 2020...\n\n"
        "Henry on 2020-01-01T00:00:04: Fireworks!\n\n"
        "Grace ðŸ§¤ ðŸ\x8d’ on 2020-01-01T00:00:05: You're a minute late!\n\n"
    )
