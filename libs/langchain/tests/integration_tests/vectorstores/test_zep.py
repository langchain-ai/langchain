# mypy: disable-error-code=attr-defined
import copy
from random import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture

from langchain.schema import Document
from langchain.vectorstores import ZepVectorStore
from langchain.vectorstores.zep import CollectionConfig

if TYPE_CHECKING:
    from zep_python.document import Document as ZepDocument
    from zep_python.document import DocumentCollection

VECTOR_DIMS = 5


def gen_vector() -> List[float]:
    return [random() for _ in range(VECTOR_DIMS)]


def gen_mock_zep_document(
    collection_name: str,
    embedding_dimensions: Optional[int] = None,
) -> "ZepDocument":
    from zep_python.document import Document as ZepDocument

    embedding = (
        [random() for _ in range(embedding_dimensions)]
        if embedding_dimensions
        else None
    )

    return ZepDocument(
        uuid=str(uuid4()),
        collection_name=collection_name,
        content="Test Document",
        embedding=embedding,
        metadata={"key": "value"},
    )


@pytest.fixture
def texts_metadatas() -> Dict[str, Any]:
    return {
        "texts": ["Test Document" for _ in range(2)],
        "metadatas": [{"key": "value"} for _ in range(2)],
    }


@pytest.fixture
def mock_documents() -> List[Document]:
    return [
        Document(
            page_content="Test Document",
            metadata={"key": "value"},
        )
        for _ in range(2)
    ]


@pytest.fixture
def texts_metadatas_as_zep_documents() -> List["ZepDocument"]:
    from zep_python.document import Document as ZepDocument

    return [
        ZepDocument(
            content="Test Document",
            metadata={"key": "value"},
        )
        for _ in range(2)
    ]


@pytest.fixture
def search_results() -> List["ZepDocument"]:
    return [
        gen_mock_zep_document(
            collection_name="test_collection", embedding_dimensions=VECTOR_DIMS
        )
        for _ in range(2)
    ]


@pytest.fixture
def search_results_with_query_embedding() -> Tuple[List["ZepDocument"], List[float]]:
    return_count = 2
    return [
        gen_mock_zep_document(
            collection_name="test_collection", embedding_dimensions=VECTOR_DIMS
        )
        for _ in range(return_count)
    ], gen_vector()


@pytest.fixture
def mock_collection_config() -> CollectionConfig:
    return CollectionConfig(
        name="test_collection",
        description="Test Collection",
        metadata={"key": "value"},
        embedding_dimensions=VECTOR_DIMS,
        is_auto_embedded=True,
    )


@pytest.fixture
@pytest.mark.requires("zep_python")
def mock_collection(
    mocker: MockerFixture,
    mock_collection_config: CollectionConfig,
    search_results: List[Document],
    search_results_with_query_embedding: Tuple[List[Document], List[float]],
) -> "DocumentCollection":
    from zep_python.document import DocumentCollection

    mock_collection: DocumentCollection = mocker.patch(
        "zep_python.document.collections.DocumentCollection", autospec=True
    )
    mock_collection.search.return_value = copy.deepcopy(search_results)
    mock_collection.asearch.return_value = copy.deepcopy(search_results)

    temp_value = copy.deepcopy(search_results_with_query_embedding)
    mock_collection.search_return_query_vector.return_value = copy.deepcopy(temp_value)
    mock_collection.asearch_return_query_vector.return_value = copy.deepcopy(temp_value)

    mock_collection.name = mock_collection_config.name
    mock_collection.is_auto_embedded = mock_collection_config.is_auto_embedded
    mock_collection.embedding_dimensions = mock_collection_config.embedding_dimensions

    return mock_collection


@pytest.fixture
@pytest.mark.requires("zep_python")
def zep_vectorstore(
    mocker: MockerFixture,
    mock_collection: "DocumentCollection",
    mock_collection_config: CollectionConfig,
) -> ZepVectorStore:
    mock_document_client = mocker.patch(
        "zep_python.document.client.DocumentClient", autospec=True
    )
    mock_document_client.get_collection.return_value = mock_collection
    mock_client = mocker.patch("zep_python.ZepClient", autospec=True)
    mock_client.return_value.document = mock_document_client

    vs = ZepVectorStore(
        mock_collection_config.name,
        "http://localhost:8080",
        api_key="test",
        config=mock_collection_config,
    )
    return vs


@pytest.mark.requires("zep_python")
def test_from_texts(
    zep_vectorstore: ZepVectorStore,
    mock_collection_config: CollectionConfig,
    mock_collection: "DocumentCollection",
    texts_metadatas: Dict[str, Any],
    texts_metadatas_as_zep_documents: List["ZepDocument"],
) -> None:
    vs = zep_vectorstore.from_texts(
        **texts_metadatas,
        collection_name=mock_collection_config.name,
        api_url="http://localhost:8000"
    )

    vs._collection.add_documents.assert_called_once_with(  # type: ignore
        texts_metadatas_as_zep_documents
    )


@pytest.mark.requires("zep_python")
def test_add_documents(
    zep_vectorstore: ZepVectorStore,
    mock_collection: "DocumentCollection",
    mock_documents: List[Document],
    texts_metadatas_as_zep_documents: List["ZepDocument"],
) -> None:
    zep_vectorstore.add_documents(mock_documents)

    mock_collection.add_documents.assert_called_once_with(  # type: ignore
        texts_metadatas_as_zep_documents
    )


@pytest.mark.requires("zep_python")
@pytest.mark.asyncio
async def test_asearch_similarity(
    zep_vectorstore: ZepVectorStore,
) -> None:
    r = await zep_vectorstore.asearch(
        query="Test Document", search_type="similarity", k=2
    )

    assert len(r) == 2
    assert r[0].page_content == "Test Document"
    assert r[0].metadata == {"key": "value"}


@pytest.mark.requires("zep_python")
@pytest.mark.asyncio
async def test_asearch_mmr(
    zep_vectorstore: ZepVectorStore,
) -> None:
    r = await zep_vectorstore.asearch(query="Test Document", search_type="mmr", k=1)

    assert len(r) == 1
    assert r[0].page_content == "Test Document"
    assert r[0].metadata == {"key": "value"}


@pytest.mark.requires("zep_python")
def test_search_similarity(
    zep_vectorstore: ZepVectorStore,
) -> None:
    r = zep_vectorstore.search(query="Test Document", search_type="similarity", k=2)

    assert len(r) == 2
    assert r[0].page_content == "Test Document"
    assert r[0].metadata == {"key": "value"}


@pytest.mark.requires("zep_python")
def test_search_mmr(
    zep_vectorstore: ZepVectorStore,
) -> None:
    r = zep_vectorstore.search(query="Test Document", search_type="mmr", k=1)

    assert len(r) == 1
    assert r[0].page_content == "Test Document"
    assert r[0].metadata == {"key": "value"}
