from __future__ import annotations

import copy
from typing import TYPE_CHECKING, List

import pytest
from pytest_mock import MockerFixture

from langchain.retrievers import ZepRetriever
from langchain.schema import Document

if TYPE_CHECKING:
    from zep_python import MemorySearchResult, ZepClient


@pytest.fixture
def search_results() -> List[MemorySearchResult]:
    from zep_python import MemorySearchResult, Message

    search_result = [
        {
            "message": {
                "uuid": "66830914-19f5-490b-8677-1ba06bcd556b",
                "created_at": "2023-05-18T20:40:42.743773Z",
                "role": "user",
                "content": "I'm looking to plan a trip to Iceland. Can you help me?",
                "token_count": 17,
            },
            "summary": None,
            "dist": 0.8734284910450115,
        },
        {
            "message": {
                "uuid": "015e618c-ba9d-45b6-95c3-77a8e611570b",
                "created_at": "2023-05-18T20:40:42.743773Z",
                "role": "user",
                "content": "How much does a trip to Iceland typically cost?",
                "token_count": 12,
            },
            "summary": None,
            "dist": 0.8554048017463456,
        },
    ]

    return [
        MemorySearchResult(
            message=Message.parse_obj(result["message"]),
            summary=result["summary"],
            dist=result["dist"],
        )
        for result in search_result
    ]


@pytest.fixture
@pytest.mark.requires("zep_python")
def zep_retriever(
    mocker: MockerFixture, search_results: List[MemorySearchResult]
) -> ZepRetriever:
    mock_zep_client: ZepClient = mocker.patch("zep_python.ZepClient", autospec=True)
    mock_zep_client.search_memory.return_value = copy.deepcopy(  # type: ignore
        search_results
    )
    mock_zep_client.asearch_memory.return_value = copy.deepcopy(  # type: ignore
        search_results
    )
    zep = ZepRetriever(session_id="123", url="http://localhost:8000")
    zep.zep_client = mock_zep_client
    return zep


@pytest.mark.requires("zep_python")
def test_zep_retriever_get_relevant_documents(
    zep_retriever: ZepRetriever, search_results: List[MemorySearchResult]
) -> None:
    documents: List[Document] = zep_retriever.get_relevant_documents(
        query="My trip to Iceland"
    )
    _test_documents(documents, search_results)


@pytest.mark.requires("zep_python")
@pytest.mark.asyncio
async def test_zep_retriever_aget_relevant_documents(
    zep_retriever: ZepRetriever, search_results: List[MemorySearchResult]
) -> None:
    documents: List[Document] = await zep_retriever.aget_relevant_documents(
        query="My trip to Iceland"
    )
    _test_documents(documents, search_results)


def _test_documents(
    documents: List[Document], search_results: List[MemorySearchResult]
) -> None:
    assert len(documents) == 2
    for i, document in enumerate(documents):
        assert document.page_content == search_results[i].message.get(  # type: ignore
            "content"
        )
        assert document.metadata.get("uuid") == search_results[
            i
        ].message.get(  # type: ignore
            "uuid"
        )
        assert document.metadata.get("role") == search_results[
            i
        ].message.get(  # type: ignore
            "role"
        )
        assert document.metadata.get("score") == search_results[i].dist
