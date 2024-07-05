import os
from pathlib import Path
from typing import Dict

import pytest
from langchain_core.documents import Document
from pytest_mock import MockerFixture

from langchain_community.document_loaders import CSVLoader, PyPDFLoader

EXAMPLE_DOCS_DIRECTORY = str(Path(__file__).parent.parent.parent / "examples/")


class MockResponse:
    def __init__(self, json_data: Dict, status_code: int):
        self.json_data = json_data
        self.status_code = status_code

    def json(self) -> Dict:
        return self.json_data


def test_pebblo_import() -> None:
    """Test that the Pebblo safe loader can be imported."""
    from langchain_community.document_loaders import PebbloSafeLoader  # noqa: F401


def test_empty_filebased_loader(mocker: MockerFixture) -> None:
    """Test basic file based csv loader."""
    # Setup
    from langchain_community.document_loaders import PebbloSafeLoader

    mocker.patch.multiple(
        "requests",
        get=MockResponse(json_data={"data": ""}, status_code=200),
        post=MockResponse(json_data={"data": ""}, status_code=200),
    )

    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "test_empty.csv")
    expected_docs: list = []

    # Exercise
    loader = PebbloSafeLoader(
        CSVLoader(file_path=file_path),
        "dummy_app_name",
        "dummy_owner",
        "dummy_description",
    )
    result = loader.load()

    # Assert
    assert result == expected_docs


def test_csv_loader_load_valid_data(mocker: MockerFixture) -> None:
    # Setup
    from langchain_community.document_loaders import PebbloSafeLoader

    mocker.patch.multiple(
        "requests",
        get=MockResponse(json_data={"data": ""}, status_code=200),
        post=MockResponse(json_data={"data": ""}, status_code=200),
    )
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "test_nominal.csv")
    full_file_path = os.path.abspath(file_path)
    expected_docs = [
        Document(
            metadata={
                "source": full_file_path,
                "row": 0,
                "full_path": full_file_path,
                "pb_id": "0",
                "content_checksum": None,  # For UT as here we are not calculating checksum
            },
            page_content="column1: value1\ncolumn2: value2\ncolumn3: value3",
        ),
        Document(
            metadata={
                "source": full_file_path,
                "row": 1,
                "full_path": full_file_path,
                "pb_id": "1",
                "content_checksum": None,  # For UT as here we are not calculating checksum
            },
            page_content="column1: value4\ncolumn2: value5\ncolumn3: value6",
        ),
    ]

    # Exercise
    loader = PebbloSafeLoader(
        CSVLoader(file_path=file_path),
        "dummy_app_name",
        "dummy_owner",
        "dummy_description",
    )
    result = loader.load()

    # Assert
    assert result == expected_docs


@pytest.mark.requires("pypdf")
def test_pdf_lazy_load(mocker: MockerFixture) -> None:
    # Setup
    from langchain_community.document_loaders import PebbloSafeLoader

    mocker.patch.multiple(
        "requests",
        get=MockResponse(json_data={"data": ""}, status_code=200),
        post=MockResponse(json_data={"data": ""}, status_code=200),
    )
    file_path = os.path.join(
        EXAMPLE_DOCS_DIRECTORY, "multi-page-forms-sample-2-page.pdf"
    )

    # Exercise
    loader = PebbloSafeLoader(
        PyPDFLoader(file_path=file_path),
        "dummy_app_name",
        "dummy_owner",
        "dummy_description",
    )

    result = list(loader.lazy_load())

    # Assert
    assert len(result) == 2


def test_pebblo_safe_loader_api_key() -> None:
    # Setup
    from langchain_community.document_loaders import PebbloSafeLoader

    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "test_empty.csv")
    api_key = "dummy_api_key"

    # Exercise
    loader = PebbloSafeLoader(
        CSVLoader(file_path=file_path),
        "dummy_app_name",
        "dummy_owner",
        "dummy_description",
        api_key=api_key,
    )

    # Assert
    assert loader.api_key == api_key
