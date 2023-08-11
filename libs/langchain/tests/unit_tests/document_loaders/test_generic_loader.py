"""Test generic loader."""
import os
import tempfile
from pathlib import Path
from typing import Generator, Iterator

import pytest

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob, FileSystemBlobLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.schema import Document


@pytest.fixture
def toy_dir() -> Generator[Path, None, None]:
    """Yield a pre-populated directory to test the blob loader."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test.txt
        with open(os.path.join(temp_dir, "test.txt"), "w") as test_txt:
            test_txt.write("This is a test.txt file.")

        # Create test.html
        with open(os.path.join(temp_dir, "test.html"), "w") as test_html:
            test_html.write(
                "<html><body><h1>This is a test.html file.</h1></body></html>"
            )

        # Create .hidden_file
        with open(os.path.join(temp_dir, ".hidden_file"), "w") as hidden_file:
            hidden_file.write("This is a hidden file.")

        # Create some_dir/nested_file.txt
        some_dir = os.path.join(temp_dir, "some_dir")
        os.makedirs(some_dir)
        with open(os.path.join(some_dir, "nested_file.txt"), "w") as nested_file:
            nested_file.write("This is a nested_file.txt file.")

        # Create some_dir/other_dir/more_nested.txt
        other_dir = os.path.join(some_dir, "other_dir")
        os.makedirs(other_dir)
        with open(os.path.join(other_dir, "more_nested.txt"), "w") as nested_file:
            nested_file.write("This is a more_nested.txt file.")

        yield Path(temp_dir)


class AsIsParser(BaseBlobParser):
    """Parser created for testing purposes."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Extract the first character of a blob."""
        yield Document(page_content=blob.as_string())


def test__init__(toy_dir: str) -> None:
    """Test initialization from init."""
    loader = GenericLoader(
        FileSystemBlobLoader(toy_dir, suffixes=[".txt"]),
        AsIsParser(),
    )
    docs = loader.load()
    assert len(docs) == 3
    # Glob order seems to be deterministic with recursion. If this test becomes flaky,
    # we can sort the docs by page content.
    assert docs[0].page_content == "This is a test.txt file."


def test_from_filesystem_classmethod(toy_dir: str) -> None:
    """Test generic loader."""
    loader = GenericLoader.from_filesystem(
        toy_dir, suffixes=[".txt"], parser=AsIsParser()
    )
    docs = loader.load()
    assert len(docs) == 3
    # Glob order seems to be deterministic with recursion. If this test becomes flaky,
    # we can sort the docs by page content.
    assert docs[0].page_content == "This is a test.txt file."


def test_from_filesystem_classmethod_with_glob(toy_dir: str) -> None:
    """Test that glob parameter is taken into account."""
    loader = GenericLoader.from_filesystem(toy_dir, glob="*.txt", parser=AsIsParser())
    docs = loader.load()
    assert len(docs) == 1
    # Glob order seems to be deterministic with recursion. If this test becomes flaky,
    # we can sort the docs by page content.
    assert docs[0].page_content == "This is a test.txt file."


@pytest.mark.requires("tqdm")
def test_from_filesystem_classmethod_show_progress(toy_dir: str) -> None:
    """Test that glob parameter is taken into account."""
    loader = GenericLoader.from_filesystem(
        toy_dir, glob="*.txt", parser=AsIsParser(), show_progress=True
    )
    docs = loader.load()
    assert len(docs) == 1
    # Glob order seems to be deterministic with recursion. If this test becomes flaky,
    # we can sort the docs by page content.
    assert docs[0].page_content == "This is a test.txt file."


def test_from_filesystem_using_default_parser(toy_dir: str) -> None:
    """Use the default generic parser."""
    loader = GenericLoader.from_filesystem(
        toy_dir,
        suffixes=[".txt"],
    )
    docs = loader.load()
    assert len(docs) == 3
    # Glob order seems to be deterministic with recursion. If this test becomes flaky,
    # we can sort the docs by page content.
    assert docs[0].page_content == "This is a test.txt file."
