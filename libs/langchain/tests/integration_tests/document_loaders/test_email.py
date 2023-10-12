from pathlib import Path

from langchain.document_loaders import OutlookMessageLoader, UnstructuredEmailLoader


def test_outlook_message_loader() -> None:
    """Test OutlookMessageLoader."""
    file_path = Path(__file__).parent.parent / "examples/hello.msg"
    loader = OutlookMessageLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["subject"] == "Test for TIF files"
    assert docs[0].metadata["sender"] == "Brian Zhou <brizhou@gmail.com>"
    assert docs[0].metadata["date"] == "Mon, 18 Nov 2013 16:26:24 +0800"
    assert docs[0].page_content == (
        "This is a test email to experiment with the MS Outlook MSG "
        "Extractor\r\n\r\n\r\n-- \r\n\r\n\r\nKind regards"
        "\r\n\r\n\r\n\r\n\r\nBrian Zhou\r\n\r\n"
    )


def test_unstructured_email_loader_with_attachments() -> None:
    file_path = Path(__file__).parent.parent / "examples/fake-email-attachment.eml"
    loader = UnstructuredEmailLoader(
        str(file_path), mode="elements", process_attachments=True
    )
    docs = loader.load()

    assert docs[-1].page_content == "Hey this is a fake attachment!"
    assert docs[-1].metadata["filename"] == "fake-attachment.txt"
    assert docs[-1].metadata["source"].endswith("fake-email-attachment.eml")
