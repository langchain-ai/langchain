from typing import Dict, List
from unittest.mock import AsyncMock, patch

import pytest

from langchain.docstore.document import Document
from langchain.document_loaders.mongodb import MongodbLoader


@pytest.fixture
def raw_docs() -> List[Dict]:
    return [
        {"_id": "1", "address": {"building": "1", "room": "1"}},
        {"_id": "2", "address": {"building": "2", "room": "2"}},
    ]


@pytest.fixture
def expected_documents() -> List[Document]:
    return [
        Document(
            page_content="{'_id': '1', 'address': {'building': '1', 'room': '1'}}",
            metadata={"database": "sample_restaurants", "collection": "restaurants"},
        ),
        Document(
            page_content="{'_id': '2', 'address': {'building': '2', 'room': '2'}}",
            metadata={"database": "sample_restaurants", "collection": "restaurants"},
        ),
    ]


def test_load_mocked(expected_documents: List[Document]) -> None:
    mock_async_load = AsyncMock()
    mock_async_load.return_value = expected_documents

    with patch(
        "langchain.document_loaders.MongodbLoader._async_load", new=mock_async_load
    ):
        loader = MongodbLoader(
            "mongodb://localhost:27017", "test_db", "test_collection"
        )

        documents = loader.load()

    assert documents == expected_documents


@pytest.mark.asyncio
@pytest.mark.requires("motor")
async def test_async_partial_load_mocked(expected_documents: List[Document]) -> None:
    loader = MongodbLoader(
        "mongodb://localhost:27017", "sample_restaurants", "restaurants"
    )
    expected_documents.remove(expected_documents[1])

    with pytest.raises(Exception) as error_PartialLoad:
        await loader._async_load()

        assert (
            str(error_PartialLoad.value)
            == "Error: Only partial collection of documents returned."
        )
