import pytest

from langchain.document_loaders import XorbitsLoader
from langchain.schema import Document

try:
    import xorbits  # noqa: F401

    xorbits_installed = True
except ImportError:
    xorbits_installed = False


@pytest.mark.skipif(not xorbits_installed, reason="xorbits not installed")
def test_load_returns_list_of_documents() -> None:
    import xorbits.pandas as pd

    data = {
        "text": ["Hello", "World"],
        "author": ["Alice", "Bob"],
        "date": ["2022-01-01", "2022-01-02"],
    }
    loader = XorbitsLoader(pd.DataFrame(data))
    docs = loader.load()
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 2


@pytest.mark.skipif(not xorbits_installed, reason="xorbits not installed")
def test_load_converts_dataframe_columns_to_document_metadata() -> None:
    import xorbits.pandas as pd

    data = {
        "text": ["Hello", "World"],
        "author": ["Alice", "Bob"],
        "date": ["2022-01-01", "2022-01-02"],
    }
    loader = XorbitsLoader(pd.DataFrame(data))
    docs = loader.load()
    expected = {
        "author": ["Alice", "Bob"],
        "date": ["2022-01-01", "2022-01-02"],
    }
    for i, doc in enumerate(docs):
        assert doc.metadata["author"] == expected["author"][i]
        assert doc.metadata["date"] == expected["date"][i]


@pytest.mark.skipif(not xorbits_installed, reason="xorbits not installed")
def test_load_uses_page_content_column_to_create_document_text() -> None:
    import xorbits.pandas as pd

    data = {
        "text": ["Hello", "World"],
        "author": ["Alice", "Bob"],
        "date": ["2022-01-01", "2022-01-02"],
    }
    sample_data_frame = pd.DataFrame(data)
    sample_data_frame = sample_data_frame.rename(columns={"text": "dummy_test_column"})
    loader = XorbitsLoader(sample_data_frame, page_content_column="dummy_test_column")
    docs = loader.load()
    assert docs[0].page_content == "Hello"
    assert docs[1].page_content == "World"
