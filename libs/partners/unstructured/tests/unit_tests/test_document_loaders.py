from pathlib import Path
from typing import Any, Callable
from unittest import mock
from unittest.mock import Mock, mock_open, patch

import pytest
from unstructured.documents.elements import Text  # type: ignore

from langchain_unstructured.document_loaders import (
    _SingleDocumentLoader,  # type: ignore
)

EXAMPLE_DOCS_DIRECTORY = str(
    Path(__file__).parent.parent.parent.parent.parent
    / "community/tests/integration_tests/examples/"
)


# --- _SingleDocumentLoader._get_content() ---


def test_it_gets_content_from_file() -> None:
    mock_file = Mock()
    mock_file.read.return_value = b"content from file"
    loader = _SingleDocumentLoader(
        client=Mock(), file=mock_file, metadata_filename="fake.txt"
    )

    content = loader._file_content  # type: ignore

    assert content == b"content from file"
    mock_file.read.assert_called_once()


@patch("builtins.open", new_callable=mock_open, read_data=b"content from file_path")
def test_it_gets_content_from_file_path(mock_file: Mock) -> None:
    loader = _SingleDocumentLoader(client=Mock(), file_path="dummy_path")

    content = loader._file_content  # type: ignore

    assert content == b"content from file_path"
    mock_file.assert_called_once_with("dummy_path", "rb")
    handle = mock_file()
    handle.read.assert_called_once()


def test_it_raises_value_error_without_file_or_file_path() -> None:
    loader = _SingleDocumentLoader(
        client=Mock(),
    )

    with pytest.raises(ValueError) as e:
        loader._file_content  # type: ignore

    assert str(e.value) == "file or file_path must be defined."


# --- _SingleDocumentLoader._elements_json ---


def test_it_calls_elements_via_api_with_valid_args() -> None:
    with patch.object(
        _SingleDocumentLoader, "_elements_via_api", new_callable=mock.PropertyMock
    ) as mock_elements_via_api:
        mock_elements_via_api.return_value = [{"element": "data"}]
        loader = _SingleDocumentLoader(
            client=Mock(),
            # Minimum required args for self._elements_via_api to be called:
            partition_via_api=True,
            api_key="some_key",
        )

        result = loader._elements_json  # type: ignore

    mock_elements_via_api.assert_called_once()
    assert result == [{"element": "data"}]


@patch.object(_SingleDocumentLoader, "_convert_elements_to_dicts")
def test_it_partitions_locally_by_default(mock_convert_elements_to_dicts: Mock) -> None:
    mock_convert_elements_to_dicts.return_value = [{}]
    with patch.object(
        _SingleDocumentLoader, "_elements_via_local", new_callable=mock.PropertyMock
    ) as mock_elements_via_local:
        mock_elements_via_local.return_value = [{}]
        # Minimum required args for self._elements_via_api to be called:
        loader = _SingleDocumentLoader(
            client=Mock(),
        )

        result = loader._elements_json  # type: ignore

    mock_elements_via_local.assert_called_once_with()
    mock_convert_elements_to_dicts.assert_called_once_with([{}])
    assert result == [{}]


def test_it_partitions_locally_and_logs_warning_with_partition_via_api_False(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with patch.object(
        _SingleDocumentLoader, "_elements_via_local"
    ) as mock_get_elements_locally:
        mock_get_elements_locally.return_value = [Text("Mock text element.")]
        loader = _SingleDocumentLoader(
            client=Mock(), partition_via_api=False, api_key="some_key"
        )

        _ = loader._elements_json  # type: ignore


# -- fixtures -------------------------------


@pytest.fixture()
def get_post_processor() -> Callable[[str], str]:
    def append_the_end(text: str) -> str:
        return text + "THE END!"

    return append_the_end


@pytest.fixture()
def fake_json_response() -> list[dict[str, Any]]:
    return [
        {
            "type": "Title",
            "element_id": "b7f58c2fd9c15949a55a62eb84e39575",
            "text": "LayoutParser: A Uniﬁed Toolkit for Deep Learning Based Document"
            "Image Analysis",
            "metadata": {
                "languages": ["eng"],
                "page_number": 1,
                "filename": "layout-parser-paper.pdf",
                "filetype": "application/pdf",
            },
        },
        {
            "type": "UncategorizedText",
            "element_id": "e1c4facddf1f2eb1d0db5be34ad0de18",
            "text": "1 2 0 2",
            "metadata": {
                "languages": ["eng"],
                "page_number": 1,
                "parent_id": "b7f58c2fd9c15949a55a62eb84e39575",
                "filename": "layout-parser-paper.pdf",
                "filetype": "application/pdf",
            },
        },
    ]


@pytest.fixture()
def fake_multiple_docs_json_response() -> list[dict[str, Any]]:
    return [
        {
            "type": "Title",
            "element_id": "b7f58c2fd9c15949a55a62eb84e39575",
            "text": "LayoutParser: A Uniﬁed Toolkit for Deep Learning Based Document"
            " Image Analysis",
            "metadata": {
                "languages": ["eng"],
                "page_number": 1,
                "filename": "layout-parser-paper.pdf",
                "filetype": "application/pdf",
            },
        },
        {
            "type": "NarrativeText",
            "element_id": "3c4ac9e7f55f1e3dbd87d3a9364642fe",
            "text": "6/29/23, 12:16\u202fam - User 4: This message was deleted",
            "metadata": {
                "filename": "whatsapp_chat.txt",
                "languages": ["eng"],
                "filetype": "text/plain",
            },
        },
    ]
