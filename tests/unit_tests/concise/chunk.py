from langchain.concise.chunk import chunk
from langchain.concise.config import get_default_text_splitter
from langchain.schema import Document


def test_chunk():

    # Test with string
    text = "The quick brown fox jumps over the lazy dog."
    result = chunk(text)
    assert isinstance(result, list)
    assert all(isinstance(chunk, str) for chunk in result)
    assert get_default_text_splitter()._merge_splits(result, " ") == text

    # Test with Document
    doc = Document(text)
    result = chunk(doc)
    assert isinstance(result, list)
    assert all(isinstance(chunk, Document) for chunk in result)
    assert get_default_text_splitter()._join_docs(result, " ") == text
