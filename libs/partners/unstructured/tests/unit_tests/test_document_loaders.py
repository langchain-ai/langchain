import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable
from unittest import mock
from unittest.mock import ANY

import pytest

from langchain_unstructured import (
    UnstructuredSDKFileIOLoader,
    UnstructuredSDKFileLoader,
)
from langchain_unstructured.document_loaders import _get_content

# EXAMPLE_DOCS_DIRECTORY = "libs/community/tests/integration_tests/examples"
EXAMPLE_DOCS_DIRECTORY = str(Path(__file__).parent.parent.parent.parent.parent / "community/tests/integration_tests/examples/")


# -- UnstructuredSDKFileLoader -------------------------------


def test_api_file_loader_calls_get_elements_from_api(fake_json_response) -> None:
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")
    loader = UnstructuredSDKFileLoader(
        file_path=file_path,
        api_key="FAKE_API_KEY",
        strategy="fast",
    )

    with mock.patch(
        "langchain_unstructured.document_loaders._get_elements_from_api",
        return_value=fake_json_response,
    ) as mock_get_elements:
        docs = loader.load()

    mock_get_elements.assert_called_once_with(
        file_path=file_path,
        api_key="FAKE_API_KEY",
        api_url="https://api.unstructuredapp.io/general/v0/general",
        strategy="fast",
    )
    assert docs[0].metadata.get("element_id") is not None
    # check that the Document metadata contains all of the element metadata
    assert all(
        docs[0].metadata[k] == v
        for k, v in fake_json_response[0].get("metadata").items()
    )


def test_api_file_loader_with_multiple_files_calls_get_elements_from_api(
    fake_multiple_docs_json_response,
) -> None:
    file_paths = [
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "whatsapp_chat.txt"),
    ]

    loader = UnstructuredSDKFileLoader(
        file_path=file_paths,
        api_key="FAKE_API_KEY",
        strategy="fast",
    )
    with mock.patch(
        "langchain_unstructured.document_loaders._get_elements_from_api",
        return_value=fake_multiple_docs_json_response,
    ) as mock_get_elements:
        docs = loader.load()

    mock_get_elements.assert_has_calls(
        [
            mock.call(
                file_path=os.path.join(
                    EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"
                ),
                api_key="FAKE_API_KEY",
                api_url="https://api.unstructuredapp.io/general/v0/general",
                strategy="fast",
            ),
            mock.call(
                file_path=os.path.join(EXAMPLE_DOCS_DIRECTORY, "whatsapp_chat.txt"),
                api_key="FAKE_API_KEY",
                api_url="https://api.unstructuredapp.io/general/v0/general",
                strategy="fast",
            ),
        ]
    )
    assert mock_get_elements.call_count == 2
    assert docs[0].metadata.get("filename") == "layout-parser-paper.pdf"
    assert docs[-1].metadata.get("filename") == "whatsapp_chat.txt"


def test_api_file_loader_ignores_mode(fake_json_response) -> None:
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")
    loader = UnstructuredSDKFileLoader(
        file_path=file_path,
        api_key="FAKE_API_KEY",
        mode="single",
        strategy="fast",
    )

    with mock.patch(
        "langchain_unstructured.document_loaders._get_elements_from_api",
        return_value=fake_json_response,
    ) as mock_get_elements:
        docs = loader.load()

    mock_get_elements.assert_called_once_with(
        file_path=file_path,
        api_key="FAKE_API_KEY",
        api_url="https://api.unstructuredapp.io/general/v0/general",
        strategy="fast",
    )
    # check the document has not been chunked into a single object
    assert (
        docs[0].page_content
        == "LayoutParser: A Uniﬁed Toolkit for Deep Learning Based"
        " DocumentImage Analysis"
    )


def test_api_file_loader_with_post_processors(
    get_post_processor, fake_json_response
) -> None:
    """Test UnstructuredAPIFileLoader._post_proceess_elements."""
    loader = UnstructuredSDKFileLoader(
        file_path=os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        api_key="FAKE_API_KEY",
        post_processors=[get_post_processor],
        strategy="fast",
    )

    with mock.patch(
        "langchain_unstructured.document_loaders._get_elements_from_api",
        return_value=fake_json_response,
    ) as mock_get_elements:
        docs = loader.load()

    mock_get_elements.assert_called_once_with(
        file_path=os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        api_key="FAKE_API_KEY",
        api_url="https://api.unstructuredapp.io/general/v0/general",
        strategy="fast",
    )
    assert docs[0].page_content.endswith("THE END!")
    assert all(
        docs[0].metadata[k] == v
        for k, v in fake_json_response[0].get("metadata").items()
    )


# -- UnstructuredSDKFileIOLoader -------------------------------


def test_api_file_io_loader_calls_get_elements_from_api(fake_json_response) -> None:
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")

    with open(file_path, "rb") as f:
        loader = UnstructuredSDKFileIOLoader(
            file=f,
            api_key="FAKE_API_KEY",
            metadata_filename=file_path,
            strategy="fast",
        )
        with mock.patch(
            "langchain_unstructured.document_loaders._get_elements_from_api",
            return_value=fake_json_response,
        ) as mock_get_elements:
            docs = loader.load()

    mock_get_elements.assert_called_once_with(
        file=ANY,
        file_path=os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        api_key="FAKE_API_KEY",
        api_url="https://api.unstructuredapp.io/general/v0/general",
        strategy="fast",
    )
    assert docs[0].metadata.get("element_id") is not None
    assert all(
        docs[0].metadata[k] == v
        for k, v in fake_json_response[0].get("metadata").items()
    )


def test_api_file_io_loader_with_multiple_files_calls_get_elements_from_api(
    fake_multiple_docs_json_response,
) -> None:
    file_paths = [
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "whatsapp_chat.txt"),
    ]

    with ExitStack() as stack:
        files = [stack.enter_context(open(file_path, "rb")) for file_path in file_paths]
        loader = UnstructuredSDKFileIOLoader(
            file=files,  # type: ignore
            api_key="FAKE_API_KEY",
            metadata_filename=file_paths,
            strategy="fast",
        )
        with mock.patch(
            "langchain_unstructured.document_loaders._get_elements_from_api",
            return_value=fake_multiple_docs_json_response,
        ) as mock_get_elements:
            docs = loader.load()

    mock_get_elements.assert_has_calls(
        [
            mock.call(
                file=ANY,
                file_path=os.path.join(
                    EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"
                ),
                api_key="FAKE_API_KEY",
                api_url="https://api.unstructuredapp.io/general/v0/general",
                strategy="fast",
            ),
            mock.call(
                file=ANY,
                file_path=os.path.join(EXAMPLE_DOCS_DIRECTORY, "whatsapp_chat.txt"),
                api_key="FAKE_API_KEY",
                api_url="https://api.unstructuredapp.io/general/v0/general",
                strategy="fast",
            ),
        ]
    )
    assert mock_get_elements.call_count == 2
    assert docs[0].metadata.get("filename") == "layout-parser-paper.pdf"  # type: ignore
    assert docs[-1].metadata.get("filename") == "whatsapp_chat.txt"  # type: ignore


def test_api_file_io_loader_with_post_processors(
    get_post_processor, fake_json_response
) -> None:
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")

    with open(file_path, "rb") as f:
        loader = UnstructuredSDKFileIOLoader(
            file=f,
            api_key="FAKE_API_KEY",
            metadata_filename=file_path,
            post_processors=[get_post_processor],
            strategy="fast",
        )
        with mock.patch(
            "langchain_unstructured.document_loaders._get_elements_from_api",
            return_value=fake_json_response,
        ) as mock_get_elements:
            docs = loader.load()

    mock_get_elements.assert_called_once_with(
        file=ANY,
        file_path=os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        api_key="FAKE_API_KEY",
        api_url="https://api.unstructuredapp.io/general/v0/general",
        strategy="fast",
    )
    assert docs[0].page_content.endswith("THE END!")
    assert all(
        docs[0].metadata[k] == v
        for k, v in fake_json_response[0].get("metadata").items()
    )


# -- _get_content() -------------------------------


def test_get_content_from_file() -> None:
    with open(
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"), "rb"
    ) as f:
        content = _get_content(
            file_path=os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
            file=f,
        )

    assert isinstance(content, bytes)
    assert content[:50] == b"%PDF-1.5\n%\x8f\n47 0 obj\n<< /Filter /FlateDecode /Leng"


def test_get_content_from_file_path() -> None:
    content = _get_content(
        file_path=os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")
    )

    assert isinstance(content, bytes)
    assert content[:50] == b"%PDF-1.5\n%\x8f\n47 0 obj\n<< /Filter /FlateDecode /Leng"


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
