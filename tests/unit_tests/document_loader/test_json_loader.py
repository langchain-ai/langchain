import pytest
from pytest import raises
from pytest_mock import MockerFixture

from langchain.docstore.document import Document
from langchain.document_loaders.json_loader import JSONLoader


@pytest.mark.requires("jq")
def test_load_valid_string_content(mocker: MockerFixture) -> None:
    file_path = "/workspaces/langchain/test.json"
    expected_docs = [
        Document(
            page_content="value1",
            metadata={"source": file_path, "seq_num": 1},
        ),
        Document(
            page_content="value2",
            metadata={"source": file_path, "seq_num": 2},
        ),
    ]
    mocker.patch("builtins.open", mocker.mock_open())
    mock_csv_reader = mocker.patch("pathlib.Path.read_text")
    mock_csv_reader.return_value = '[{"text": "value1"}, {"text": "value2"}]'

    loader = JSONLoader(file_path=file_path, jq_schema=".[].text", text_content=True)
    result = loader.load()

    assert result == expected_docs


@pytest.mark.requires("jq")
def test_load_valid_dict_content(mocker: MockerFixture) -> None:
    file_path = "/workspaces/langchain/test.json"
    expected_docs = [
        Document(
            page_content='{"text": "value1"}',
            metadata={"source": file_path, "seq_num": 1},
        ),
        Document(
            page_content='{"text": "value2"}',
            metadata={"source": file_path, "seq_num": 2},
        ),
    ]
    mocker.patch("builtins.open", mocker.mock_open())
    mock_csv_reader = mocker.patch("pathlib.Path.read_text")
    mock_csv_reader.return_value = """
        [{"text": "value1"}, {"text": "value2"}]
    """

    loader = JSONLoader(file_path=file_path, jq_schema=".[]", text_content=False)
    result = loader.load()

    assert result == expected_docs


@pytest.mark.requires("jq")
def test_load_valid_bool_content(mocker: MockerFixture) -> None:
    file_path = "/workspaces/langchain/test.json"
    expected_docs = [
        Document(
            page_content="False",
            metadata={"source": file_path, "seq_num": 1},
        ),
        Document(
            page_content="True",
            metadata={"source": file_path, "seq_num": 2},
        ),
    ]
    mocker.patch("builtins.open", mocker.mock_open())
    mock_csv_reader = mocker.patch("pathlib.Path.read_text")
    mock_csv_reader.return_value = """
        [
            {"flag": false}, {"flag": true}
        ]
    """

    loader = JSONLoader(file_path=file_path, jq_schema=".[].flag", text_content=False)
    result = loader.load()

    assert result == expected_docs


@pytest.mark.requires("jq")
def test_load_valid_numeric_content(mocker: MockerFixture) -> None:
    file_path = "/workspaces/langchain/test.json"
    expected_docs = [
        Document(
            page_content="99",
            metadata={"source": file_path, "seq_num": 1},
        ),
        Document(
            page_content="99.5",
            metadata={"source": file_path, "seq_num": 2},
        ),
    ]
    mocker.patch("builtins.open", mocker.mock_open())
    mock_csv_reader = mocker.patch("pathlib.Path.read_text")
    mock_csv_reader.return_value = """
        [
            {"num": 99}, {"num": 99.5}
        ]
    """

    loader = JSONLoader(file_path=file_path, jq_schema=".[].num", text_content=False)
    result = loader.load()

    assert result == expected_docs


@pytest.mark.requires("jq")
def test_load_invalid_test_content(mocker: MockerFixture) -> None:
    file_path = "/workspaces/langchain/test.json"
    mocker.patch("builtins.open", mocker.mock_open())
    mock_csv_reader = mocker.patch("pathlib.Path.read_text")
    mock_csv_reader.return_value = """
        [{"text": "value1"}, {"text": "value2"}]
    """

    loader = JSONLoader(file_path=file_path, jq_schema=".[]", text_content=True)

    with raises(ValueError):
        loader.load()
