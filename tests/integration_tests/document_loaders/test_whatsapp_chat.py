from pathlib import Path

from langchain.document_loaders import WhatsAppChatLoader


def test_whatsapp_chat_loader() -> None:
    """Test WhatsAppChatLoader."""
    file_path = Path(__file__).parent.parent / "examples" / "whatsapp_chat.txt"
    loader = WhatsAppChatLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["source"] == str(file_path)
    assert docs[0].page_content == (
        "James on 05.05.23, 15:48:11: Hi here\n\n"
        "User name on 11/8/21, 9:41:32 AM: Message 123\n\n"
        "User 2 on 1/23/23, 3:19 AM: Bye!\n\n"
        "User 1 on 1/23/23, 3:22_AM: And let me know if anything changes\n\n"
        "~ User name 2 on 1/24/21, 12:41:03 PM: Of course!\n\n"
    )
