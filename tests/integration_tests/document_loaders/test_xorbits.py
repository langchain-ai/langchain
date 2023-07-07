from langchain.document_loaders import XorbitsLoader
from langchain.schema import Document
import pytest

@pytest.fixture
def sample_data_frame():
    import xorbits.pandas as pd
    data = {
        "text": ["Hello", "World"],
        "author": ["Alice", "Bob"],
        "date": ["2022-01-01", "2022-01-02"],
    }
    return pd.DataFrame(data)


def test_load_returns_list_of_documents(sample_data_frame) -> None:
    loader = XorbitsLoader(sample_data_frame)
    docs = loader.load()
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 2


def test_load_converts_dataframe_columns_to_document_metadata(
    sample_data_frame,
) -> None:
    loader = XorbitsLoader(sample_data_frame)
    docs = loader.load()
    expected = {"author": ["Alice", "Bob"],
                "date": ["2022-01-01", "2022-01-02"],}
    for i, doc in enumerate(docs):
        assert doc.metadata["author"] == expected["author"][i]
        assert doc.metadata["date"] == expected["date"][i]


def test_load_uses_page_content_column_to_create_document_text(
    sample_data_frame,
) -> None:
    sample_data_frame = sample_data_frame.rename(columns={"text": "dummy_test_column"})
    loader = XorbitsLoader(sample_data_frame, page_content_column="dummy_test_column")
    docs = loader.load()
    assert docs[0].page_content == "Hello"
    assert docs[1].page_content == "World"
