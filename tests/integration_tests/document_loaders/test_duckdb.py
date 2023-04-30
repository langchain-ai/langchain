import unittest

from langchain.document_loaders.duckdb_loader import DuckDBLoader

try:
    import duckdb  # noqa: F401

    duckdb_installed = True
except ImportError:
    duckdb_installed = False


@unittest.skipIf(not duckdb_installed, "duckdb not installed")
def test_duckdb_loader_no_options() -> None:
    """Test DuckDB loader."""

    loader = DuckDBLoader("SELECT 1 AS a, 2 AS b")
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1\nb: 2"
    assert docs[0].metadata == {}


@unittest.skipIf(not duckdb_installed, "duckdb not installed")
def test_duckdb_loader_page_content_columns() -> None:
    """Test DuckDB loader."""

    loader = DuckDBLoader(
        "SELECT 1 AS a, 2 AS b UNION SELECT 3 AS a, 4 AS b",
        page_content_columns=["a"],
    )
    docs = loader.load()

    assert len(docs) == 2
    assert docs[0].page_content == "a: 1"
    assert docs[0].metadata == {}

    assert docs[1].page_content == "a: 3"
    assert docs[1].metadata == {}


@unittest.skipIf(not duckdb_installed, "duckdb not installed")
def test_duckdb_loader_metadata_columns() -> None:
    """Test DuckDB loader."""

    loader = DuckDBLoader(
        "SELECT 1 AS a, 2 AS b",
        page_content_columns=["a"],
        metadata_columns=["b"],
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1"
    assert docs[0].metadata == {"b": 2}
