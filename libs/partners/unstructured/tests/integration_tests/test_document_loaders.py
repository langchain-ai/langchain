import os
from pathlib import Path
from typing import Callable
from unittest import mock

import pytest

from langchain_unstructured import UnstructuredLoader

EXAMPLE_DOCS_DIRECTORY = str(Path(__file__).parent.parent.parent.parent.parent / "community/tests/integration_tests/examples/")


# -- Local partition --


def test_loader_partitions_locally() -> None:
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")
    docs = UnstructuredLoader(
        file_path=file_path,
        strategy="fast",
        # Unstructured kwargs
        include_page_breaks=True,
    ).load()
    assert all(doc.metadata.get("filename") == "layout-parser-paper.pdf" for doc in docs)
    assert any(doc.metadata.get("category") == "PageBreak" for doc in docs)


def test_loader_partitions_locally_and_applies_post_processors(get_post_processor):
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")
    loader = UnstructuredLoader(
        file_path=file_path,
        post_processors=[get_post_processor],
        strategy="fast",
    )
    docs = loader.load()

    assert len(docs) > 1
    assert docs[0].page_content.endswith("THE END!")


# -- API partition --


def test_loader_calls_elements_via_api(FAKE_JSON_RESPONSE) -> None:
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")
    loader = UnstructuredLoader(
        file_path=file_path,
        api_key="FAKE_API_KEY",
        partition_via_api=True,
        strategy="fast",
    )

    with mock.patch.object(
        UnstructuredLoader, "_elements_via_api", new_callable=mock.PropertyMock,
    ) as mock_client:
        mock_client.return_value = FAKE_JSON_RESP
        docs = loader.load()

    assert docs[0].metadata.get("element_id") is not None
    # check that the Document metadata contains all of the element metadata
    assert all(
        docs[0].metadata[k] == v
        for k, v in FAKE_JSON_RESPONSE[0].get("metadata").items()
    )


# -- fixtures and constants -------------------------------


@pytest.fixture()
def get_post_processor() -> Callable[[str], str]:
    def append_the_end(text: str) -> str:
        return text + "THE END!"

    return append_the_end


FAKE_JSON_RESP = [
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


FAKE_MULTIPLE_DOCS_JSON_RESP = [
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
