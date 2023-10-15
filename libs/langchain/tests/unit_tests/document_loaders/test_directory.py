from pathlib import Path

import pytest

from langchain.document_loaders import DirectoryLoader, TextLoader

TEST_DOCS_DIR = Path(__file__).parent / "sample_documents"


def test_raise_error_if_path_not_exist() -> None:
    loader = DirectoryLoader("./not_exist_directory")
    with pytest.raises(FileNotFoundError) as e:
        loader.load()

    assert str(e.value) == "Directory not found: './not_exist_directory'"


def test_raise_error_if_path_is_not_directory() -> None:
    loader = DirectoryLoader(__file__)
    with pytest.raises(ValueError) as e:
        loader.load()

    assert str(e.value) == f"Expected directory, got file: '{__file__}'"


def test_glob_with_exclude() -> None:
    loader = DirectoryLoader(
        path=str(TEST_DOCS_DIR / "directory_glob"),
        glob="**/*.txt",
        exclude_glob="**/exclude.txt",
        # DirectoryLoader uses UnstructuredTextLoader by default,
        # replace it with TextLoader to make the test runnable
        # without installing `unstructured`.
        loader_cls=TextLoader,
    )
    documents = loader.load()

    # Two include.txt files in the directory
    assert len(documents) == 2
