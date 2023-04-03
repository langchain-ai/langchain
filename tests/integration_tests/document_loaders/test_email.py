from pathlib import Path

from langchain.document_loaders import OutlookMessageLoader


def test_outlook_message_loader() -> None:
    """Test OutlookMessageLoader."""
    file_path = Path(__file__).parent.parent / "examples/Hello World.msg"
    loader = OutlookMessageLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["subject"] == "Hello World"
    assert docs[0].metadata["sender"] == '"Masand, Sahil" <Sahil.Masand@cbh.com.au>'
    assert docs[0].metadata["date"] == "Mon, 03 Apr 2023 15:18:38 +0800"
    assert docs[0].page_content == "This is a test message.\r\n\r\n"
