import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import pytest

from langchain_community.document_loaders import (
    UnstructuredAPIFileIOLoader,
    UnstructuredAPIFileLoader,
    UnstructuredFileIOLoader,
    UnstructuredFileLoader,
)
from langchain_community.document_loaders.unstructured import _get_content

EXAMPLE_DOCS_DIRECTORY = str(Path(__file__).parent.parent / "examples/")


# -- UnstructuredFileLoader -------------------------------


def test_unstructured_file_loader_with_multiple_files() -> None:
    """Test unstructured loader."""
    file_paths = [
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "whatsapp_chat.txt"),
    ]

    loader = UnstructuredFileLoader(
        file_path=file_paths,
        mode="elements",
        strategy="fast",
    )
    docs = loader.load()

    assert docs[0].metadata.get("element_id") is not None
    assert docs[0].metadata.get("filename") == "layout-parser-paper.pdf"
    assert docs[-1].metadata.get("filename") == "whatsapp_chat.txt"


def test_unstructured_file_loader_with_post_processor(get_post_processor) -> None:
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")
    loader = UnstructuredFileLoader(
        file_path=file_path,
        post_processors=[get_post_processor],
        strategy="fast",
        mode="elements",
    )
    docs = loader.load()

    assert len(docs) > 1
    assert docs[0].page_content.endswith("THE END!")


# -- UnstructuredAPIFileLoader -------------------------------


def test_unstructured_api_file_loader(json_response) -> None:
    """Test unstructured loader."""

    loader = UnstructuredAPIFileLoader(
        file_path=os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        api_key="FAKE_API_KEY",
        mode="elements",
        strategy="fast",
    )

    with mock.patch(
        "langchain_community.document_loaders.unstructured.get_elements_from_api",
        return_value=json_response,
    ) as mock_get_elements:
        docs = loader.load()

    assert mock_get_elements.assert_called_once
    assert docs[0].metadata.get("element_id") is not None
    assert docs[0].metadata.get("metadata") == json_response[0].get("metadata")


def test_unstructured_api_file_loader_with_multiple_files(
    multiple_docs_json_response,
) -> None:
    """Test unstructured loader."""
    file_paths = [
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "whatsapp_chat.txt"),
    ]

    loader = UnstructuredAPIFileLoader(
        file_path=file_paths,
        api_key="FAKE_API_KEY",
        mode="elements",
        strategy="fast",
    )
    with mock.patch(
        "langchain_community.document_loaders.unstructured.UnstructuredAPIFileLoader._get_elements",
        return_value=multiple_docs_json_response,
    ) as mock_get_elements:
        docs = loader.load()

    assert mock_get_elements.assert_called_once
    assert docs[0].metadata.get("metadata").get("filename") == "layout-parser-paper.pdf"  # type: ignore
    assert docs[-1].metadata.get("metadata").get("filename") == "whatsapp_chat.txt"  # type: ignore


def test_unstructured_api_file_loader_with_post_processors(
    get_post_processor, json_response
) -> None:
    """Test UnstructuredAPIFileLoader._post_proceess_elements."""
    loader = UnstructuredAPIFileLoader(
        file_path=os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        api_key="FAKE_API_KEY",
        mode="elements",
        post_processors=[get_post_processor],
        strategy="fast",
    )

    with mock.patch(
        "langchain_community.document_loaders.unstructured.get_elements_from_api",
        return_value=json_response,
    ) as mock_get_elements:
        docs = loader.load()

    assert mock_get_elements.assert_called_once
    assert docs[0].page_content.endswith("THE END!")
    assert docs[0].metadata.get("metadata") == json_response[0].get("metadata")


# -- UnstructuredFileIOLoader -------------------------------


def test_unstructured_file_io_loader() -> None:
    """Test unstructured loader."""
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")

    with open(file_path, "rb") as f:
        loader = UnstructuredFileIOLoader(
            file=f,
            mode="elements",
            strategy="fast",
        )
        docs = loader.load()

    assert docs[0].metadata.get("element_id") is not None
    assert len(docs) > 1


# -- UnstructuredAPIFileIOLoader -------------------------------


def test_unstructured_api_file_io_loader(json_response) -> None:
    """Test unstructured loader."""
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")

    with open(file_path, "rb") as f:
        loader = UnstructuredAPIFileIOLoader(
            file=f,
            api_key="FAKE_API_KEY",
            metadata_filename=file_path,
            mode="elements",
            strategy="fast",
        )
        with mock.patch(
            "langchain_community.document_loaders.unstructured.get_elements_from_api",
            return_value=json_response,
        ) as mock_get_elements:
            docs = loader.load()

    assert mock_get_elements.assert_called_once
    assert docs[0].metadata.get("element_id") is not None
    assert docs[0].metadata.get("metadata") == json_response[0].get("metadata")


def test_unstructured_api_file_io_loader_with_multiple_files(
    multiple_docs_json_response,
) -> None:
    """Test unstructured loader."""
    file_paths = [
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "whatsapp_chat.txt"),
    ]

    with ExitStack() as stack:
        files = [stack.enter_context(open(file_path, "rb")) for file_path in file_paths]
        loader = UnstructuredAPIFileIOLoader(
            file=files,  # type: ignore
            api_key="FAKE_API_KEY",
            metadata_filename=file_paths,
            mode="elements",
            strategy="fast",
        )
        with mock.patch(
            "langchain_community.document_loaders.unstructured.UnstructuredAPIFileIOLoader._get_elements",
            return_value=multiple_docs_json_response,
        ) as mock_get_elements:
            docs = loader.load()

    assert mock_get_elements.assert_called_once
    assert docs[0].metadata.get("metadata").get("filename") == "layout-parser-paper.pdf"  # type: ignore
    assert docs[-1].metadata.get("metadata").get("filename") == "whatsapp_chat.txt"  # type: ignore


def test_unstructured_api_file_io_loader_with_post_processors(
    get_post_processor, json_response
) -> None:
    """Test unstructured loader."""
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")

    with open(file_path, "rb") as f:
        loader = UnstructuredAPIFileIOLoader(
            file=f,
            api_key="FAKE_API_KEY",
            metadata_filename=file_path,
            mode="elements",
            post_processors=[get_post_processor],
            strategy="fast",
        )
        with mock.patch(
            "langchain_community.document_loaders.unstructured.get_elements_from_api",
            return_value=json_response,
        ) as mock_get_elements:
            docs = loader.load()

    assert mock_get_elements.assert_called_once
    assert docs[0].page_content.endswith("THE END!")
    assert docs[0].metadata.get("metadata") == json_response[0].get("metadata")


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
def json_response() -> list[dict[str, Any]]:
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
def multiple_docs_json_response() -> list[dict[str, Any]]:
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
