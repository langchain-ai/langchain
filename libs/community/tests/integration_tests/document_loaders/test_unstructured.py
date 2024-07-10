import os
from contextlib import ExitStack
from pathlib import Path
from typing import Callable
from unittest import mock
from unittest.mock import ANY

import pytest

from unstructured.documents.elements import Element
from unstructured.staging.base import dict_to_elements

from langchain_community.document_loaders import (
    UnstructuredAPIFileIOLoader,
    UnstructuredAPIFileLoader,
    UnstructuredFileIOLoader,
    UnstructuredFileLoader,
)

EXAMPLE_DOCS_DIRECTORY = str(Path(__file__).parent.parent / "examples/")


# -- UnstructuredFileLoader -------------------------------

def test_FileLoader_with_multiple_files() -> None:
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


def test_FileLoader_with_post_processor(get_post_processor) -> None:
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

def test_APIFileLoader_calls_get_elements_from_api(fake_element_response) -> None:
    """Test unstructured loader."""
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")
    loader = UnstructuredAPIFileLoader(
        file_path=file_path,
        api_key="FAKE_API_KEY",
        mode="elements",
        strategy="fast",
    )

    with mock.patch(
        "langchain_community.document_loaders.unstructured._get_elements_from_api",
        return_value=fake_element_response,
    ) as mock_get_elements:
        docs = loader.load()
    
    # .assert_called_once_with() was failing, but the following works
    mock_get_elements.assert_has_calls([
        mock.call(
            file_path=file_path,
            api_key='FAKE_API_KEY',
            api_url='https://api.unstructuredapp.io/general/v0/general',
            strategy='fast',
        )
    ])
    assert docs[0].metadata.get("element_id") is not None
    assert set(
        fake_element_response[0].metadata.to_dict().keys()
    ).issubset(
        set(docs[0].metadata.keys())
    )


def test_APIFileLoader_with_multiple_files_calls_get_elements_from_api(
    fake_multiple_docs_element_response,
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
        "langchain_community.document_loaders.unstructured._get_elements_from_api",
        return_value=fake_multiple_docs_element_response,
    ) as mock_get_elements:
        docs = loader.load()

    mock_get_elements.assert_called_once_with(
        file_path=file_paths,
        api_key='FAKE_API_KEY',
        api_url='https://api.unstructuredapp.io/general/v0/general',
        strategy='fast',
    )
    assert docs[0].metadata.get("filename") == "layout-parser-paper.pdf"
    assert docs[-1].metadata.get("filename") == "whatsapp_chat.txt"
    assert set(
        fake_multiple_docs_element_response[0].metadata.to_dict().keys()
    ).issubset(
        set(docs[0].metadata.keys())
    )


def test_APIFileLoader_with_post_processors(
    get_post_processor, fake_element_response
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
        "langchain_community.document_loaders.unstructured._get_elements_from_api",
        return_value=fake_element_response,
    ) as mock_get_elements:
        docs = loader.load()

    mock_get_elements.assert_called_once_with(
        file_path=os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        api_key='FAKE_API_KEY',
        api_url='https://api.unstructuredapp.io/general/v0/general',
        strategy='fast',
    )
    assert docs[0].page_content.endswith("THE END!")
    assert set(
        fake_element_response[0].metadata.to_dict().keys()
    ).issubset(
        set(docs[0].metadata.keys())
    )


# -- UnstructuredFileIOLoader -------------------------------

def test_FileIOLoader() -> None:
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

def test_APIFileIOLoader_calls_get_elements_from_api(fake_element_response) -> None:
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
            "langchain_community.document_loaders.unstructured._get_elements_from_api",
            return_value=fake_element_response,
        ) as mock_get_elements:
            docs = loader.load()

    mock_get_elements.assert_called_once_with(
        file=ANY,
        file_path=os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        api_key='FAKE_API_KEY',
        api_url='https://api.unstructuredapp.io/general/v0/general',
        strategy='fast',
    )
    assert docs[0].metadata.get("element_id") is not None
    assert set(
        fake_element_response[0].metadata.to_dict().keys()
    ).issubset(
        set(docs[0].metadata.keys())
    )


def test_APIFileIOLoader_with_multiple_files_calls_get_elements_from_api(
    fake_multiple_docs_element_response,
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
            "langchain_community.document_loaders.unstructured._get_elements_from_api",
            return_value=fake_multiple_docs_element_response,
        ) as mock_get_elements:
            docs = loader.load()

    mock_get_elements.assert_called_once_with(
        file=ANY,
        file_path=file_paths,
        api_key='FAKE_API_KEY',
        api_url='https://api.unstructuredapp.io/general/v0/general',
        strategy='fast',
    )
    assert docs[0].metadata.get("filename") == "layout-parser-paper.pdf"
    assert docs[-1].metadata.get("filename") == "whatsapp_chat.txt"
    assert set(
        fake_multiple_docs_element_response[0].metadata.to_dict().keys()
    ).issubset(
        set(docs[0].metadata.keys())
    )


def test_APIFileIOLoader_with_post_processors(
    get_post_processor, fake_element_response
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
            "langchain_community.document_loaders.unstructured._get_elements_from_api",
            return_value=fake_element_response,
        ) as mock_get_elements:
            docs = loader.load()

    mock_get_elements.assert_called_once_with(
        file=ANY,
        file_path=os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        api_key='FAKE_API_KEY',
        api_url='https://api.unstructuredapp.io/general/v0/general',
        strategy='fast',
    )
    assert docs[0].page_content.endswith("THE END!")
    assert set(
        fake_element_response[0].metadata.to_dict().keys()
    ).issubset(
        set(docs[0].metadata.keys())
    )


# -- fixtures -------------------------------

@pytest.fixture()
def get_post_processor() -> Callable[[str], str]:
    def append_the_end(text: str) -> str:
        return text + "THE END!"

    return append_the_end


@pytest.fixture()
def fake_element_response() -> list[Element]:
    return dict_to_elements([
        {
            'type': 'Title', 
            'element_id': 'b7f58c2fd9c15949a55a62eb84e39575',
            'text': 'LayoutParser: A Uniﬁed Toolkit for Deep Learning Based Document Image Analysis',
            'metadata': {
                'languages': ['eng'],
                'page_number': 1,
                'filename': 'layout-parser-paper.pdf',
                'filetype': 'application/pdf'
            }
        },
        {
            'type': 'Title',
            'element_id': 'ef4798cd9f5754511eafa712c6143ac6',
            'text': 'layout analysis.',
            'metadata': {
                'languages': ['eng'],
                'page_number': 16,
                'filename': 'layout-parser-paper.pdf',
                'filetype': 'application/pdf'
            }
        }
    ])


@pytest.fixture()
def fake_multiple_docs_element_response() -> list[Element]:
    return dict_to_elements([
        {
            'type': 'Title', 
            'element_id': 'b7f58c2fd9c15949a55a62eb84e39575',
            'text': 'LayoutParser: A Uniﬁed Toolkit for Deep Learning Based Document Image Analysis',
            'metadata': {
                'languages': ['eng'],
                'page_number': 1,
                'filename': 'layout-parser-paper.pdf',
                'filetype': 'application/pdf'
            }
        },
        {
            'type': 'NarrativeText',
            'element_id': '3c4ac9e7f55f1e3dbd87d3a9364642fe',
            'text': '6/29/23, 12:16\u202fam - User 4: This message was deleted',
            'metadata': {
                'filename': 'whatsapp_chat.txt',
                'languages': ['eng'],
                'filetype': 'text/plain'
            }
        }
    ])
