from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain.docstore.document import Document
from langchain.document_loaders import MongodbLoader


@pytest.fixture
def raw_docs():
    return [
        {"_id": "1", "address": {"building": "1", "room": "1"}},
        {"_id": "2", "address": {"building": "2", "room": "2"}},
    ]


@pytest.fixture
def expected_documents():
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


@pytest.fixture
def mock_mongodb(mocker, raw_docs):
    try:
        from motor.motor_asyncio import AsyncIOMotorClient # type: ignore

        try:

            async def mock_find():
                for doc in raw_docs:
                    yield doc

            mock_collection = MagicMock()
            mock_db = MagicMock()
            mock_client = MagicMock()

            mocker.patch.object(mock_collection, "find", return_value=mock_find())
            mocker.patch.object(
                AsyncIOMotorClient, "get_database", return_value=mock_db
            )
            mocker.patch.object(mock_db, "get_collection", return_value=mock_collection)
            mocker.patch(
                "motor.motor_asyncio.AsyncIOMotorClient", return_value=mock_client
            )
        except Exception as e:
            raise ValueError("Could not mock motor. ") from e

    except ImportError:
        raise ValueError(
            "Could not import AsyncIOMotorClient package. "
            "Please install it with `pip install motor`."
        )
    return mocker


@pytest.mark.asyncio
async def test_async_load_mocked(mock_mongodb, expected_documents):
    loader = MongodbLoader(
        "mongodb://localhost:27017", "sample_restaurants", "restaurants"
    )
    documents = await loader._async_load()
    assert documents == expected_documents


@pytest.mark.asyncio
async def test_async_partial_load_mocked(expected_documents):
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


def test_load_mocked(expected_documents):
    mock_async_load = AsyncMock()
    mock_async_load.return_value = expected_documents

    with patch("langchain.document_loaders.MongodbLoader", MagicMock()):
        loader = MongodbLoader(
            "mongodb://localhost:27017", "test_db", "test_collection"
        )

    loader._async_load = mock_async_load

    documents = loader.load()

    assert documents == expected_documents
