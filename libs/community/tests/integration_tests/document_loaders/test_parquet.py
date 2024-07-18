import os
from pathlib import Path

import pytest

from langchain_community.document_loaders import ParquetLoader

EXAMPLE_DIRECTORY = file_path = Path(__file__).parent.parent / "examples"
PARQUET_FILE: str = "mlb_teams_2012.parquet"


@pytest.fixture
@pytest.mark.requires("pyarrow")
def doc_list():
    """Fixture to load the document list from a Parquet file."""
    file_path = os.path.join(EXAMPLE_DIRECTORY, PARQUET_FILE)
    loader = ParquetLoader(file_path=str(file_path), content_columns=["Team"])
    return loader.load()


def test_parquetloader_list_length(doc_list):
    assert len(doc_list) == 30


def test_parquetloader_content(doc_list):
    for doc in doc_list:
        assert "Payroll" in doc.metadata.keys()
        assert "Wins" in doc.metadata.keys()
        assert doc.page_content is not None
