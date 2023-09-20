from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.mark.requires("motor")
@pytest.mark.asyncio
async def test_load_mocked(expected_documents: List[Document]) -> None:
    mock_async_load = AsyncMock()
    mock_async_load.return_value = expected_documents

    mock_find = AsyncMock()
    mock_find.return_value = iter(expected_documents)

    mock_count_documents = MagicMock()
    mock_count_documents.return_value = len(expected_documents)

    mock_collection = MagicMock()
    mock_collection.find = mock_find
    mock_collection.count_documents = mock_count_documents

    with patch(
        "motor.motor_asyncio.AsyncIOMotorClient", return_value=MagicMock()
    ), patch(
        "langchain.document_loaders.mongodb.MongodbLoader.aload",
        new=mock_async_load,
    ):
        loader = MongodbLoader(
            "mongodb://localhost:27017", "test_db", "test_collection"
        )
        loader.collection = mock_collection
        documents = await loader.aload()

    assert documents == expected_documents
