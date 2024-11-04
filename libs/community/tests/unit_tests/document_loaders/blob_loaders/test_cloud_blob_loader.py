"""Verify that file system blob loader works as expected."""

import os
import tempfile
from typing import Generator
from urllib.parse import urlparse

import pytest

from langchain_community.document_loaders.blob_loaders import CloudBlobLoader


@pytest.fixture
def toy_dir() -> Generator[str, None, None]:
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

        yield f"file://{temp_dir}"


# @pytest.fixture
# @pytest.mark.requires("boto3")
# def toy_dir() -> str:
#     return "s3://ppr-langchain-test"


_TEST_CASES = [
    {
        "glob": "**/[!.]*",
        "suffixes": None,
        "exclude": (),
        "relative_filenames": [
            "test.html",
            "test.txt",
            "some_dir/nested_file.txt",
            "some_dir/other_dir/more_nested.txt",
        ],
    },
    {
        "glob": "*",
        "suffixes": None,
        "exclude": (),
        "relative_filenames": ["test.html", "test.txt", ".hidden_file"],
    },
    {
        "glob": "**/*.html",
        "suffixes": None,
        "exclude": (),
        "relative_filenames": ["test.html"],
    },
    {
        "glob": "*/*.txt",
        "suffixes": None,
        "exclude": (),
        "relative_filenames": ["some_dir/nested_file.txt"],
    },
    {
        "glob": "**/*.txt",
        "suffixes": None,
        "exclude": (),
        "relative_filenames": [
            "test.txt",
            "some_dir/nested_file.txt",
            "some_dir/other_dir/more_nested.txt",
        ],
    },
    {
        "glob": "**/*",
        "suffixes": [".txt"],
        "exclude": (),
        "relative_filenames": [
            "test.txt",
            "some_dir/nested_file.txt",
            "some_dir/other_dir/more_nested.txt",
        ],
    },
    {
        "glob": "meeeeeeow",
        "suffixes": None,
        "exclude": (),
        "relative_filenames": [],
    },
    {
        "glob": "*",
        "suffixes": [".html", ".txt"],
        "exclude": (),
        "relative_filenames": ["test.html", "test.txt"],
    },
    # Using exclude patterns
    {
        "glob": "**/*",
        "suffixes": [".txt"],
        "exclude": ("some_dir/*",),
        "relative_filenames": ["test.txt", "some_dir/other_dir/more_nested.txt"],
    },
    # Using 2 exclude patterns, one of which is recursive
    {
        "glob": "**/*",
        "suffixes": None,
        "exclude": ("**/*.txt", ".hidden*"),
        "relative_filenames": ["test.html"],
    },
]


@pytest.mark.requires("cloudpathlib")
@pytest.mark.parametrize("params", _TEST_CASES)
def test_file_names_exist(toy_dir: str, params: dict) -> None:
    """Verify that the file names exist."""

    glob_pattern = params["glob"]
    suffixes = params["suffixes"]
    exclude = params["exclude"]
    relative_filenames = params["relative_filenames"]

    loader = CloudBlobLoader(
        toy_dir, glob=glob_pattern, suffixes=suffixes, exclude=exclude
    )
    blobs = list(loader.yield_blobs())

    url_parsed = urlparse(toy_dir)
    scheme = ""
    if url_parsed.scheme == "file":
        scheme = "file://"

    file_names = sorted(f"{scheme}{blob.path}" for blob in blobs)

    expected_filenames = sorted(
        str(toy_dir + "/" + relative_filename)
        for relative_filename in relative_filenames
    )

    assert file_names == expected_filenames
    assert loader.count_matching_files() == len(relative_filenames)


@pytest.mark.requires("cloudpathlib")
def test_show_progress(toy_dir: str) -> None:
    """Verify that file system loader works with a progress bar."""
    loader = CloudBlobLoader(toy_dir)
    blobs = list(loader.yield_blobs())
    assert len(blobs) == loader.count_matching_files()
