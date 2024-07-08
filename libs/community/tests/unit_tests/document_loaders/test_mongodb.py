from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders.mongodb import MongodbLoader


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


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_db() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_collection() -> MagicMock:
    return MagicMock()


@pytest.fixture
def loader(mock_client: MagicMock, mock_db: MagicMock, 
           mock_collection: MagicMock) -> MongodbLoader:
    mock_client.get_database.return_value = mock_db
    mock_db.get_collection.return_value = mock_collection

    with patch(
        "langchain_community.document_loaders.mongodb.AsyncIOMotorClient",
        return_value=mock_client,
    ):
        return MongodbLoader(
            connection_string="mongodb://localhost:27017",
            db_name="sample_restaurants",
            collection_name="restaurants",
        )


@pytest.mark.requires("motor")
def test_constructor(loader: MongodbLoader) -> None:
    assert loader.db_name == "sample_restaurants"
    assert loader.collection_name == "restaurants"


@pytest.mark.requires("motor")
async def test_aload(
    mock_collection: MagicMock, 
    loader: MongodbLoader, 
    raw_docs: List[Dict], 
    expected_documents: List[Document]
) -> None:
    mock_collection.count_documents.return_value = len(raw_docs)
    mock_collection.find.return_value = iter(raw_docs)

    documents = await loader.aload()
    assert len(documents) == len(expected_documents)
    for doc, expected_doc in zip(documents, expected_documents):
        assert doc.page_content == expected_doc.page_content
        assert doc.metadata == expected_doc.metadata


@pytest.mark.requires("motor")
def test_construct_projection(loader: MongodbLoader) -> None:
    loader.field_names = ["address"]
    loader.metadata_names = ["_id"]
    expected_projection = {"address": 1, "_id": 1}
    projection = loader._construct_projection()
    assert projection == expected_projection


@pytest.mark.requires("motor")
async def test_load_method(
    mock_collection: MagicMock, 
    loader: MongodbLoader, 
    raw_docs: List[Dict], 
    expected_documents: List[Document]
) -> None:
    mock_collection.count_documents.return_value = len(raw_docs)
    mock_collection.find.return_value = iter(raw_docs)

    documents = loader.load()
    assert len(documents) == len(expected_documents)
    for doc, expected_doc in zip(documents, expected_documents):
        assert doc.page_content == expected_doc.page_content
        assert doc.metadata == expected_doc.metadata


@pytest.mark.requires("motor")
async def test_filter_criteria(
    mock_collection: MagicMock, 
    raw_docs: List[Dict]
) -> None:
    mock_client = MagicMock()
    mock_client.get_database.return_value = MagicMock()
    mock_db = mock_client.get_database()
    mock_db.get_collection.return_value = mock_collection

    filter_criteria = {"address.building": "1"}
    loader = MongodbLoader(
        connection_string="mongodb://localhost:27017",
        db_name="sample_restaurants",
        collection_name="restaurants",
        filter_criteria=filter_criteria,
    )

    mock_collection.count_documents.return_value = 1
    mock_collection.find.return_value = iter([raw_docs[0]])

    documents = await loader.aload()
    assert len(documents) == 1
    assert documents[0].page_content == str(raw_docs[0])


@pytest.mark.requires("motor")
async def test_include_db_collection_in_metadata(
    mock_collection: MagicMock, 
    raw_docs: List[Dict]
) -> None:
    mock_client = MagicMock()
    mock_client.get_database.return_value = MagicMock()
    mock_db = mock_client.get_database()
    mock_db.get_collection.return_value = mock_collection

    loader = MongodbLoader(
        connection_string="mongodb://localhost:27017",
        db_name="sample_restaurants",
        collection_name="restaurants",
        include_db_collection_in_metadata=True,
    )

    mock_collection.count_documents.return_value = len(raw_docs)
    mock_collection.find.return_value = iter(raw_docs)

    documents = await loader.aload()
    assert len(documents) == len(raw_docs)
    for doc in documents:
        assert doc.metadata["database"] == "sample_restaurants"
        assert doc.metadata["collection"] == "restaurants"
