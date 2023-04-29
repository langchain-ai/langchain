from pytest_mock import MockerFixture

from langchain.docstore.document import Document
from langchain.document_loaders.csv_loader import CSVLoader


class TestCSVLoader:
    # Tests that a CSV file with valid data is loaded successfully.
    def test_csv_loader_load_valid_data(self, mocker: MockerFixture) -> None:
        # Setup
        file_path = "test.csv"
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
        mocker.patch("builtins.open", mocker.mock_open())
        mock_csv_reader = mocker.patch("csv.DictReader")
        mock_csv_reader.return_value = [
            {"column1": "value1", "column2": "value2", "column3": "value3"},
            {"column1": "value4", "column2": "value5", "column3": "value6"},
        ]

        # Exercise
        loader = CSVLoader(file_path=file_path)
        result = loader.load()

        # Assert
        assert result == expected_docs

    # Tests that an empty CSV file is handled correctly.
    def test_csv_loader_load_empty_file(self, mocker: MockerFixture) -> None:
        # Setup
        file_path = "test.csv"
        expected_docs: list = []
        mocker.patch("builtins.open", mocker.mock_open())
        mock_csv_reader = mocker.patch("csv.DictReader")
        mock_csv_reader.return_value = []

        # Exercise
        loader = CSVLoader(file_path=file_path)
        result = loader.load()

        # Assert
        assert result == expected_docs

    # Tests that a CSV file with only one row is handled correctly.
    def test_csv_loader_load_single_row_file(self, mocker: MockerFixture) -> None:
        # Setup
        file_path = "test.csv"
        expected_docs = [
            Document(
                page_content="column1: value1\ncolumn2: value2\ncolumn3: value3",
                metadata={"source": file_path, "row": 0},
            )
        ]
        mocker.patch("builtins.open", mocker.mock_open())
        mock_csv_reader = mocker.patch("csv.DictReader")
        mock_csv_reader.return_value = [
            {"column1": "value1", "column2": "value2", "column3": "value3"}
        ]

        # Exercise
        loader = CSVLoader(file_path=file_path)
        result = loader.load()

        # Assert
        assert result == expected_docs

    # Tests that a CSV file with only one column is handled correctly.
    def test_csv_loader_load_single_column_file(self, mocker: MockerFixture) -> None:
        # Setup
        file_path = "test.csv"
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
        mocker.patch("builtins.open", mocker.mock_open())
        mock_csv_reader = mocker.patch("csv.DictReader")
        mock_csv_reader.return_value = [
            {"column1": "value1"},
            {"column1": "value2"},
            {"column1": "value3"},
        ]

        # Exercise
        loader = CSVLoader(file_path=file_path)
        result = loader.load()

        # Assert
        assert result == expected_docs
