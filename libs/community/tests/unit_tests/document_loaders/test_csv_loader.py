from pathlib import Path

import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders.csv_loader import CSVLoader


class TestCSVLoader:
    # Tests that a CSV file with valid data is loaded successfully.
    def test_csv_loader_load_valid_data(self) -> None:
        # Setup
        file_path = self._get_csv_file_path("test_nominal.csv")
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

        # Exercise
        loader = CSVLoader(file_path=file_path)
        result = loader.load()

        # Assert
        assert result == expected_docs

    # Tests that an empty CSV file is handled correctly.
    def test_csv_loader_load_empty_file(self) -> None:
        # Setup
        file_path = self._get_csv_file_path("test_empty.csv")
        expected_docs: list = []

        # Exercise
        loader = CSVLoader(file_path=file_path)
        result = loader.load()

        # Assert
        assert result == expected_docs

    # Tests that a CSV file with only one row is handled correctly.
    def test_csv_loader_load_single_row_file(self) -> None:
        # Setup
        file_path = self._get_csv_file_path("test_one_row.csv")
        expected_docs = [
            Document(
                page_content="column1: value1\ncolumn2: value2\ncolumn3: value3",
                metadata={"source": file_path, "row": 0},
            )
        ]

        # Exercise
        loader = CSVLoader(file_path=file_path)
        result = loader.load()

        # Assert
        assert result == expected_docs

    # Tests that a CSV file with only one column is handled correctly.
    def test_csv_loader_load_single_column_file(self) -> None:
        # Setup
        file_path = self._get_csv_file_path("test_one_col.csv")
        expected_docs = [
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

        # Exercise
        loader = CSVLoader(file_path=file_path)
        result = loader.load()

        # Assert
        assert result == expected_docs

    def test_csv_loader_load_none_column_file(self) -> None:
        # Setup
        file_path = self._get_csv_file_path("test_none_col.csv")
        expected_docs = [
            Document(
                page_content="column1: value1\ncolumn2: value2\n"
                "column3: value3\nNone: value4,value5",
                metadata={"source": file_path, "row": 0},
            ),
            Document(
                page_content="column1: value6\ncolumn2: value7\n"
                "column3: value8\nNone: value9",
                metadata={"source": file_path, "row": 1},
            ),
        ]

        # Exercise
        loader = CSVLoader(file_path=file_path)
        result = loader.load()

        # Assert
        assert result == expected_docs

    def test_csv_loader_content_columns(self) -> None:
        # Setup
        file_path = self._get_csv_file_path("test_none_col.csv")
        expected_docs = [
            Document(
                page_content="column1: value1\n" "column3: value3",
                metadata={"source": file_path, "row": 0},
            ),
            Document(
                page_content="column1: value6\n" "column3: value8",
                metadata={"source": file_path, "row": 1},
            ),
        ]

        # Exercise
        loader = CSVLoader(file_path=file_path, content_columns=("column1", "column3"))
        result = loader.load()

        # Assert
        assert result == expected_docs

    def test_csv_loader_raise(self) -> None:
        # Setup

        file_path = self._get_csv_file_path("test_raise.csv")
        missing_column = "missing_column"

        with pytest.raises(
            ValueError,
            match=f"Source column '{missing_column}' not found in CSV file.",
        ):
            # Exercise
            loader = CSVLoader(
                file_path=file_path,
                source_column=missing_column,
            )
            loader.load()

        with pytest.raises(
            ValueError,
            match=f"Metadata column '{missing_column}' not found in CSV file.",
        ):
            # Exercise
            loader = CSVLoader(
                file_path=file_path,
                metadata_columns=(missing_column, "colum1", "colum2", "colum3"),
            )
            loader.load()

    # utility functions
    def _get_csv_file_path(self, file_name: str) -> str:
        return str(Path(__file__).resolve().parent / "test_docs" / "csv" / file_name)
