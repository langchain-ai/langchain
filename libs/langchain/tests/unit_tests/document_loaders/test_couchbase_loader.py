from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from langchain.docstore.document import Document
from langchain.document_loaders.couchbase import CouchbaseLoader


def test_couchbase_import() -> None:
    """Test that the Couchbase document loader can be imported."""
    from langchain.document_loaders import CouchbaseLoader  # noqa: F401


@pytest.fixture
def sample_docs() -> List[Dict]:
    return [
        {
            "id": 1,
            "name": "Sample",
            "address": {"street": "A", "zip": 12345},
        },
        {
            "id": 2,
            "name": "Demo",
            "address": {"street": "B", "zip": 54321},
        },
    ]


@pytest.fixture
def expected_documents() -> List[Document]:
    return [
        Document(
            page_content="address: {'street': 'A', 'zip': 12345}\nid: 1\nname: Sample",
            metadata={"id": 1},
        ),
        Document(
            page_content="address: {'street': 'B', 'zip': 54321}\nid: 2\nname: Demo",
            metadata={"id": 2},
        ),
    ]


@pytest.mark.requires("couchbase")
def test_load(expected_documents: List[Document]) -> None:
    mock_load = MagicMock()
    mock_load.return_value = expected_documents

    with patch(
        "langchain.document_loaders.couchbase.CouchbaseLoader.load", new=mock_load
    ):
        loader = CouchbaseLoader(
            "couchbase://localhost",
            "Administrator",
            "Password",
            query="select d.* from docs d",
            metadata_fields=["id"],
        )
        documents = loader.load()
    assert documents == expected_documents
