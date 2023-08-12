import copy
from random import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture

from langchain.schema import Document
from langchain.vectorstores import ZepVectorStore

if TYPE_CHECKING:
    from zep_python.document import Document as ZepDocument
    from zep_python.document import DocumentCollection


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
            collection_name="test_collection", embedding_dimensions=10
        )
        for _ in range(2)
    ]


@pytest.fixture
def search_results_with_query_embedding() -> Tuple[List["ZepDocument"], List[float]]:
    return_count = 2
    return [
        gen_mock_zep_document(
            collection_name="test_collection", embedding_dimensions=10
        )
        for _ in range(return_count)
    ], [random() for _ in range(return_count)]


@pytest.fixture
@pytest.mark.requires("zep_python")
def mock_collection(
    mocker: MockerFixture,
    search_results: List[Document],
    search_results_with_query_embedding: Tuple[List[Document], List[float]],
) -> "DocumentCollection":
    from zep_python.document import DocumentCollection

    mock_collection: DocumentCollection = mocker.patch(
        "zep_python.document.collections.DocumentCollection", autospec=True
    )
    mock_collection.search.return_value = copy.deepcopy(search_results)  # type: ignore
    mock_collection.asearch.return_value = copy.deepcopy(search_results)  # type: ignore
    mock_collection.search_return_query_vector.return_value = copy.deepcopy(  # type: ignore
        search_results_with_query_embedding
    )
    mock_collection.asearch_return_query_vector.return_value = copy.deepcopy(  # type: ignore
        search_results_with_query_embedding
    )
    mock_collection.is_auto_embedded = True
    return mock_collection


@pytest.fixture
@pytest.mark.requires("zep_python")
def zep_vectorstore(
    mocker: MockerFixture,
    mock_collection: "DocumentCollection",
) -> ZepVectorStore:
    vs = ZepVectorStore(collection=mock_collection)
    return vs


@pytest.mark.requires("zep_python")
def test_from_texts(
    zep_vectorstore: ZepVectorStore,
    mock_collection: "DocumentCollection",
    texts_metadatas: texts_metadatas,
    texts_metadatas_as_zep_documents: List["ZepDocument"],
) -> None:
    vs = zep_vectorstore.from_texts(**texts_metadatas, collection=mock_collection)

    mock_collection.add_documents.assert_called_once_with( # type: ignore
        texts_metadatas_as_zep_documents
    )
