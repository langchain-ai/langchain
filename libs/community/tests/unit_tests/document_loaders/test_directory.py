from pathlib import Path
from typing import Any, Iterator, List

import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader


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


class CustomLoader(TextLoader):
    """Test loader. Mimics interface of existing file loader."""

    def __init__(self, path: Path, **kwargs: Any) -> None:
        """Initialize the loader."""
        self.path = path

    def load(self) -> List[Document]:
        """Load documents."""
        with open(self.path, "r") as f:
            return [Document(page_content=f.read())]

    def lazy_load(self) -> Iterator[Document]:
        raise NotImplementedError("CustomLoader does not implement lazy_load()")


def test_exclude_ignores_matching_files(tmp_path: Path) -> None:
    txt_file = tmp_path / "test.txt"
    py_file = tmp_path / "test.py"
    txt_file.touch()
    py_file.touch()
    loader = DirectoryLoader(
        str(tmp_path),
        exclude=["*.py"],
        loader_cls=CustomLoader,  # type: ignore
    )
    data = loader.load()
    assert len(data) == 1


def test_exclude_as_string_converts_to_sequence() -> None:
    loader = DirectoryLoader("./some_directory", exclude="*.py")
    assert loader.exclude == ("*.py",)


class CustomLoaderMetadataOnly(CustomLoader):
    """Test loader that just returns the file path in metadata. For test_directory_loader_glob_multiple."""  # noqa: E501

    def load(self) -> List[Document]:
        metadata = {"source": self.path}
        return [Document(page_content="", metadata=metadata)]

    def lazy_load(self) -> Iterator[Document]:
        return iter(self.load())


def test_directory_loader_glob_multiple() -> None:
    """Verify that globbing multiple patterns in a list works correctly."""

    path_to_examples = "tests/examples/"
    list_extensions = [".rst", ".txt"]
    list_globs = [f"**/*{ext}" for ext in list_extensions]
    is_file_type_loaded = {ext: False for ext in list_extensions}

    loader = DirectoryLoader(
        path=path_to_examples, glob=list_globs, loader_cls=CustomLoaderMetadataOnly
    )

    list_documents = loader.load()

    for doc in list_documents:
        path_doc = Path(doc.metadata.get("source", ""))
        ext_doc = path_doc.suffix

        if is_file_type_loaded.get(ext_doc, False):
            continue
        elif ext_doc in list_extensions:
            is_file_type_loaded[ext_doc] = True
        else:
            # Loaded a filetype that was not specified in extensions list
            assert False

    for ext in list_extensions:
        assert is_file_type_loaded.get(ext, False)
