from unittest import mock

import pytest

from langchain.docstore.document import Document
from langchain.document_loaders import PandocRSTLoader


def test_load_valid_file() -> None:
    loader = PandocRSTLoader("test.rst")

    with mock.patch("os.path.isfile") as mock_isfile, mock.patch(
        "pypandoc.convert_file"
    ) as mock_convert:
        mock_isfile.return_value = True
        mock_convert.return_value = "converted markdown content"

        documents = loader.load()
        document = documents[0]

        assert isinstance(document, Document)
        assert document.page_content == "converted markdown content"


def test_load_file_not_exists() -> None:
    loader = PandocRSTLoader("test.rst")

    with mock.patch("os.path.isfile") as mock_isfile:
        mock_isfile.return_value = False

        with pytest.raises(FileNotFoundError):
            loader.load()


def test_load_pypandoc_not_installed() -> None:
    loader = PandocRSTLoader("test.rst")

    with mock.patch("os.path.isfile") as mock_isfile, mock.patch(
        "pypandoc.convert_file", side_effect=ImportError
    ):
        mock_isfile.return_value = True

        with pytest.raises(ImportError):
            loader.load()
