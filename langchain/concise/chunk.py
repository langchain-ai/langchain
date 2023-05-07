from langchain.concise import config
from langchain.schema import Document


def chunk(text: str | Document) -> list[str] | list[Document]:
    """Split text into chunks."""
    if isinstance(text, Document):
        return config.get_default_text_splitter().split_document(text)
    elif isinstance(text, str):
        return config.get_default_text_splitter().split_text(text)
    else:
        raise TypeError(f"Expected str or Document, got {type(text)}")
