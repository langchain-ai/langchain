import uuid
from typing import Dict

import pytest
from pytest_mock import MockerFixture

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.retrievers import RemoteLangChainRetriever
from langchain.schema import Document


class MockResponse:
    def __init__(self, json_data: Dict, status_code: int):
        self.json_data = json_data
        self.status_code = status_code

    def json(self) -> Dict:
        return self.json_data


def mocked_requests_post() -> MockResponse:
    return MockResponse(
        json_data={
            "message": [
                {
                    "page_content": "I like apples",
                    "metadata": {
                        "test": 0,
                    },
                },
                {
                    "page_content": "I like pineapples",
                    "metadata": {
                        "test": 1,
                    },
                },
            ]
        },
        status_code=200,
    )


@pytest.fixture
def test_RemoteLangChainRetriever__get_relevant_documents(
    mocker: MockerFixture,
) -> None:
    mocker.patch("requests.post", side_effect=mocked_requests_post)

    remote_langchain_retriever = RemoteLangChainRetriever(
        url="http://localhost:8000",
    )
    response = remote_langchain_retriever._get_relevant_documents(
        query="I like apples",
        run_manager=CallbackManagerForRetrieverRun(
            run_id=uuid.uuid4(),
            handlers=[],
            inheritable_handlers=[],
        ),
    )
    want = [
        Document(page_content="I like apples", metadata={"test": 0}),
        Document(page_content="I like pineapples", metadata={"test": 1}),
    ]

    assert len(response) == len(want)
    for r, w in zip(response, want):
        assert r.page_content == w.page_content
        assert r.metadata == w.metadata


# TODO: _aget_relevant_documents test
