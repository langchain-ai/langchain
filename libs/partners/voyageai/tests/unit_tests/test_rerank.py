from collections import namedtuple
from typing import Any

import pytest  # type: ignore
from langchain_core.documents import Document
from voyageai.api_resources import VoyageResponse  # type: ignore
from voyageai.object import RerankingObject  # type: ignore

from langchain_voyageai.rerank import VoyageAIRerank


@pytest.mark.requires("voyageai")
def test_init() -> None:
    VoyageAIRerank(
        voyageai_api_key="foo",
        model="rerank-lite-1",
    )


@pytest.mark.requires("voyageai")
def test_rerank_unit_test(mocker: Any) -> None:
    expected_result = [
        Document(
            page_content="Photosynthesis in plants converts light energy into "
            "glucose and produces essential oxygen.",
            metadata={"relevance_score": 0.9},
        ),
        Document(
            page_content="The Mediterranean diet emphasizes fish, olive oil, and "
            "vegetables, believed to reduce chronic diseases.",
            metadata={"relevance_score": 0.8},
        ),
    ]

    VoyageResultItem = namedtuple("VoyageResultItem", ["index", "relevance_score"])
    Usage = namedtuple("Usage", ["prompt_tokens", "total_tokens", "completion_tokens"])

    doc_list = [
        "The Mediterranean diet emphasizes fish, olive oil, and vegetables"
        ", believed to reduce chronic diseases.",
        "Photosynthesis in plants converts light energy into glucose and "
        "produces essential oxygen.",
        "20th-century innovations, from radios to smartphones, centered "
        "on electronic advancements.",
        "Rivers provide water, irrigation, and habitat for aquatic species, "
        "vital for ecosystems.",
        "Apple’s conference call to discuss fourth fiscal quarter results and "
        "business updates is scheduled for Thursday, November 2, 2023 at 2:00 "
        "p.m. PT / 5:00 p.m. ET.",
        "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' "
        "endure in literature.",
    ]
    documents = [Document(page_content=x) for x in doc_list]

    voyage_response = VoyageResponse()
    voyage_response.data = [
        VoyageResultItem(index=1, relevance_score=0.9),
        VoyageResultItem(index=0, relevance_score=0.8),
    ]
    voyage_response.usage = Usage(
        prompt_tokens=0, total_tokens=255, completion_tokens=0
    )
    mock_result = RerankingObject(response=voyage_response, documents=["X", "Y"])

    rerank = VoyageAIRerank(
        voyageai_api_key="foo",
        model="rerank-lite-1",
    )
    mocker.patch("voyageai.Client.rerank").return_value = mock_result
    result = rerank.compress_documents(
        documents=documents, query="When is the Apple's conference call scheduled?"
    )
    assert expected_result == result
