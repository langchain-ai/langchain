import pandas as pd
import pytest

from langchain.document_loaders import DataFrameLoader
from langchain.schema import Document


@pytest.fixture
def sample_data_frame() -> pd.DataFrame:
    data = {
        "text": ["Hello", "World"],
        "author": ["Alice", "Bob"],
        "date": ["2022-01-01", "2022-01-02"],
    }
    return pd.DataFrame(data)


def test_load_returns_list_of_documents(sample_data_frame: pd.DataFrame) -> None:
    loader = DataFrameLoader(sample_data_frame)
    docs = loader.load()
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 2


def test_load_converts_dataframe_columns_to_document_metadata(
    sample_data_frame: pd.DataFrame,
) -> None:
    loader = DataFrameLoader(sample_data_frame)
    docs = loader.load()
    for i, doc in enumerate(docs):
        assert doc.metadata["author"] == sample_data_frame.loc[i, "author"]
        assert doc.metadata["date"] == sample_data_frame.loc[i, "date"]


def test_load_uses_page_content_column_to_create_document_text(
    sample_data_frame: pd.DataFrame,
) -> None:
    sample_data_frame = sample_data_frame.rename(columns={"text": "dummy_test_column"})
    loader = DataFrameLoader(sample_data_frame, page_content_column="dummy_test_column")
    docs = loader.load()
    assert docs[0].page_content == "Hello"
    assert docs[1].page_content == "World"
