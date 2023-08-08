import pytest

from langchain.document_loaders.bigquery import BigQueryLoader

try:
    from google.cloud import bigquery  # noqa: F401

    bigquery_installed = True
except ImportError:
    bigquery_installed = False


@pytest.mark.skipif(not bigquery_installed, reason="bigquery not installed")
def test_bigquery_loader_no_options() -> None:
    loader = BigQueryLoader("SELECT 1 AS a, 2 AS b")
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1\nb: 2"
    assert docs[0].metadata == {}


@pytest.mark.skipif(not bigquery_installed, reason="bigquery not installed")
def test_bigquery_loader_page_content_columns() -> None:
    loader = BigQueryLoader(
        "SELECT 1 AS a, 2 AS b UNION ALL SELECT 3 AS a, 4 AS b",
        page_content_columns=["a"],
    )
    docs = loader.load()

    assert len(docs) == 2
    assert docs[0].page_content == "a: 1"
    assert docs[0].metadata == {}

    assert docs[1].page_content == "a: 3"
    assert docs[1].metadata == {}


@pytest.mark.skipif(not bigquery_installed, reason="bigquery not installed")
def test_bigquery_loader_metadata_columns() -> None:
    loader = BigQueryLoader(
        "SELECT 1 AS a, 2 AS b",
        page_content_columns=["a"],
        metadata_columns=["b"],
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1"
    assert docs[0].metadata == {"b": 2}
