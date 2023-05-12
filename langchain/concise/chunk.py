from langchain.concise import config
from langchain.schema import Document
from langchain.text_splitter import TextSplitter


def chunk(
    text: str | Document, text_splitter: TextSplitter = None
) -> list[str] | list[Document]:
    """Split text into chunks."""
    text_splitter = text_splitter or config.get_default_text_splitter()
    if isinstance(text, Document):
        return text_splitter.split_document(text)
    elif isinstance(text, str):
        return text_splitter.split_text(text)
    else:
        raise TypeError(f"Expected str or Document, got {type(text)}")
