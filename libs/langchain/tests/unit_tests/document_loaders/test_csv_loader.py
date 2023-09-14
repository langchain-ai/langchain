from pathlib import Path

from langchain.docstore.document import Document
from langchain.document_loaders.csv_loader import CSVLoader


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

    # utility functions
    def _get_csv_file_path(self, file_name: str) -> str:
        return str(Path(__file__).resolve().parent / "test_docs" / "csv" / file_name)
