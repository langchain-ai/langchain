from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from langchain.document_loaders import PolarsDataFrameLoader
from langchain.schema import Document

if TYPE_CHECKING:
    import polars as pl


@pytest.fixture
def sample_data_frame() -> pl.DataFrame:
    import polars as pl

    data = {
        "text": ["Hello", "World"],
        "author": ["Alice", "Bob"],
        "date": ["2022-01-01", "2022-01-02"],
    }
    return pl.DataFrame(data)


def test_load_returns_list_of_documents(sample_data_frame: pl.DataFrame) -> None:
    loader = PolarsDataFrameLoader(sample_data_frame)
    docs = loader.load()
    assert isinstance(docs, list)
    assert all(isinstance(doc, Document) for doc in docs)
    assert len(docs) == 2


def test_load_converts_dataframe_columns_to_document_metadata(
    sample_data_frame: pl.DataFrame,
) -> None:
    import polars as pl

    loader = PolarsDataFrameLoader(sample_data_frame)
    docs = loader.load()

    for i, doc in enumerate(docs):
        df: pl.DataFrame = sample_data_frame[i]
        assert df is not None
        assert doc.metadata["author"] == df.select("author").item()
        assert doc.metadata["date"] == df.select("date").item()


def test_load_uses_page_content_column_to_create_document_text(
    sample_data_frame: pl.DataFrame,
) -> None:
    sample_data_frame = sample_data_frame.rename(mapping={"text": "dummy_test_column"})
    loader = PolarsDataFrameLoader(
        sample_data_frame, page_content_column="dummy_test_column"
    )
    docs = loader.load()
    assert docs[0].page_content == "Hello"
    assert docs[1].page_content == "World"
