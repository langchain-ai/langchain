from pathlib import Path

import pytest

from langchain_community.document_loaders import DirectoryLoader


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


@pytest.mark.requires("unstructured")
def test_exclude_ignores_matching_files(tmp_path: Path) -> None:
    txt_file = tmp_path / "test.txt"
    py_file = tmp_path / "test.py"
    txt_file.touch()
    py_file.touch()

    loader = DirectoryLoader(str(tmp_path), exclude=["*.py"])
    data = loader.load()

    assert len(data) == 1


def test_exclude_as_string_converts_to_sequence() -> None:
    loader = DirectoryLoader("./some_directory", exclude="*.py")
    assert loader.exclude == ("*.py",)
