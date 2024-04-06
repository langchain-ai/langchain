from pathlib import Path

import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.directory import DirectoryLoader


class TestDirectoryLoader:
    # Tests that lazy loading a CSV file with multiple documents is successful.
    def test_directory_loader_lazy_load_single_file_multiple_docs(self) -> None:
        # Setup
        dir_path = self._get_csv_dir_path()
        file_name = "test_nominal.csv"
        file_path = self._get_csv_file_path(file_name)
        expected_docs = [
            Document(
                page_content="column1: value1\ncolumn2: value2\ncolumn3: value3",
                metadata={"source": file_path, "row": 0},
            ),
            Document(
                page_content="column1: value4\ncolumn2: value5\ncolumn3: value6",
                metadata={"source": file_path, "row": 1},
            ),
        ]

        # Assert
        loader = DirectoryLoader(dir_path, glob=file_name, loader_cls=CSVLoader)
        for i, doc in enumerate(loader.lazy_load()):
            assert doc == expected_docs[i]

    # Tests that lazy loading an empty CSV file is handled correctly.
    def test_directory_loader_lazy_load_empty_file(self) -> None:
        # Setup
        dir_path = self._get_csv_dir_path()
        file_name = "test_empty.csv"

        # Assert
        loader = DirectoryLoader(dir_path, glob=file_name, loader_cls=CSVLoader)
        for _ in loader.lazy_load():
            pytest.fail(
                "DirectoryLoader.lazy_load should not yield something for an empty file"
            )

    # Tests that lazy loading multiple CSV files is handled correctly.
    def test_directory_loader_lazy_load_multiple_files(self) -> None:
        # Setup
        dir_path = self._get_csv_dir_path()
        file_name = "test_nominal.csv"
        file_path = self._get_csv_file_path(file_name)
        expected_docs = [
            Document(
                page_content="column1: value1\ncolumn2: value2\ncolumn3: value3",
                metadata={"source": file_path, "row": 0},
            ),
            Document(
                page_content="column1: value4\ncolumn2: value5\ncolumn3: value6",
                metadata={"source": file_path, "row": 1},
            ),
        ]
        file_name = "test_one_col.csv"
        file_path = self._get_csv_file_path(file_name)
        expected_docs += [
            Document(
                page_content="column1: value1",
                metadata={"source": file_path, "row": 0},
            ),
            Document(
                page_content="column1: value2",
                metadata={"source": file_path, "row": 1},
            ),
            Document(
                page_content="column1: value3",
                metadata={"source": file_path, "row": 2},
            ),
        ]
        file_name = "test_one_row.csv"
        file_path = self._get_csv_file_path(file_name)
        expected_docs += [
            Document(
                page_content="column1: value1\ncolumn2: value2\ncolumn3: value3",
                metadata={"source": file_path, "row": 0},
            )
        ]

        # Assert
        loader = DirectoryLoader(dir_path, loader_cls=CSVLoader)
        loaded_docs = []
        for doc in loader.lazy_load():
            assert doc in expected_docs
            loaded_docs.append(doc)
        assert len(loaded_docs) == len(expected_docs)

    # utility functions
    def _get_csv_file_path(self, file_name: str) -> str:
        return str(Path(__file__).resolve().parent / "test_docs" / "csv" / file_name)

    def _get_csv_dir_path(self) -> str:
        return str(Path(__file__).resolve().parent / "test_docs" / "csv")
