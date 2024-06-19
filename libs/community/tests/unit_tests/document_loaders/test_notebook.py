import json

from pytest_mock import MockerFixture

from langchain_community.document_loaders.notebook import NotebookLoader


def test_initialization() -> None:
    loader = NotebookLoader(path="./testfile.ipynb")
    assert loader.file_path == "./testfile.ipynb"


def test_load_no_outputs(mocker: MockerFixture) -> None:
    mock_notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Test notebook\n", "This is a test notebook."],
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": ["Hello World!\n"],
                    }
                ],
            }
        ]
    }
    mocked_cell_type = mock_notebook_content["cells"][0]["cell_type"]
    mocked_source = mock_notebook_content["cells"][0]["source"]

    # Convert the mock notebook content to a JSON string
    mock_notebook_content_str = json.dumps(mock_notebook_content)

    # Mock the open function & json.load functions
    mocker.patch("builtins.open", mocker.mock_open(read_data=mock_notebook_content_str))
    mocker.patch("json.load", return_value=mock_notebook_content)

    loader = NotebookLoader(path="./testfile.ipynb")
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == f"'{mocked_cell_type}' cell: '{mocked_source}'\n\n"
    assert docs[0].metadata == {"source": "testfile.ipynb"}


def test_load_with_outputs(mocker: MockerFixture) -> None:
    mock_notebook_content: dict = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["# Test notebook\n", "This is a test notebook."],
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": ["Hello World!\n"],
                    }
                ],
            }
        ]
    }
    mocked_cell_type = mock_notebook_content["cells"][0]["cell_type"]
    mocked_source = mock_notebook_content["cells"][0]["source"]
    mocked_output = mock_notebook_content["cells"][0]["outputs"][0]["text"]

    # Convert the mock notebook content to a JSON string
    mock_notebook_content_str = json.dumps(mock_notebook_content)

    # Mock the open function & json.load functions
    mocker.patch("builtins.open", mocker.mock_open(read_data=mock_notebook_content_str))
    mocker.patch("json.load", return_value=mock_notebook_content)

    loader = NotebookLoader(path="./testfile.ipynb", include_outputs=True)
    docs = loader.load()

    assert len(docs) == 1
    expected_content = (
        f"'{mocked_cell_type}' cell: '{mocked_source}'\n"
        f" with output: '{mocked_output}'\n\n"
    )
    assert docs[0].page_content == expected_content
    assert docs[0].metadata == {"source": "testfile.ipynb"}
