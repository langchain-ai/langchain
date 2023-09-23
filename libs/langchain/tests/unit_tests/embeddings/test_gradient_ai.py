from typing import Dict

import pytest
from pytest_mock import MockerFixture

from langchain.embeddings import GradientEmbeddings

_MODEL_ID = "my_model_valid_id"
_GRADIENT_SECRET = "secret_valid_token_123456"
_GRADIENT_WORKSPACE_ID = "valid_workspace_12345"
_GRADIENT_BASE_URL = "https://api.gradient.ai/api"
_DOCUMENTS = [
    "pizza",
    "another pizza",
    "a document",
    "another pizza",
    "super long document with many tokens",
]


class MockResponse:
    def __init__(self, json_data: Dict, status_code: int):
        self.json_data = json_data
        self.status_code = status_code

    def json(self) -> Dict:
        return self.json_data


def mocked_requests_post(
    url: str,
    headers: dict,
    json: dict,
) -> MockResponse:
    assert url.startswith(_GRADIENT_BASE_URL)
    assert _MODEL_ID in url
    assert json
    assert headers

    assert headers.get("authorization") == f"Bearer {_GRADIENT_SECRET}"
    assert headers.get("x-gradient-workspace-id") == f"{_GRADIENT_WORKSPACE_ID}"

    assert "inputs" in json and "input" in json["inputs"][0]
    embeddings = []
    for inp in json["inputs"]:
        # verify correct ordering
        inp = inp["input"]
        if "pizza" in inp:
            v = [1.0, 0.0, 0.0]
        elif "document" in inp:
            v = [0.0, 0.9, 0.0]
        else:
            v = [0.0, 0.0, -1.0]
        if len(inp) > 10:
            v[2] += 0.1
        embeddings.append({"embedding": v})

    return MockResponse(
        json_data={"embeddings": embeddings},
        status_code=200,
    )


def test_gradient_llm_sync(
    mocker: MockerFixture,
) -> None:
    mocker.patch("requests.post", side_effect=mocked_requests_post)

    embedder = GradientEmbeddings(
        gradient_api_url=_GRADIENT_BASE_URL,
        gradient_access_token=_GRADIENT_SECRET,
        gradient_workspace_id=_GRADIENT_WORKSPACE_ID,
        model=_MODEL_ID,
    )
    assert embedder.gradient_access_token == _GRADIENT_SECRET
    assert embedder.gradient_api_url == _GRADIENT_BASE_URL
    assert embedder.gradient_workspace_id == _GRADIENT_WORKSPACE_ID
    assert embedder.model == _MODEL_ID

    response = embedder.embed_documents(_DOCUMENTS)
    want = [
        [1.0, 0.0, 0.0],  # pizza
        [1.0, 0.0, 0.1],  # pizza  + long
        [0.0, 0.9, 0.0],  # doc
        [1.0, 0.0, 0.1],  # pizza + long
        [0.0, 0.9, 0.1],  # doc + long
    ]

    assert response == want


def test_gradient_llm_large_batch_size(
    mocker: MockerFixture,
) -> None:
    mocker.patch("requests.post", side_effect=mocked_requests_post)

    embedder = GradientEmbeddings(
        gradient_api_url=_GRADIENT_BASE_URL,
        gradient_access_token=_GRADIENT_SECRET,
        gradient_workspace_id=_GRADIENT_WORKSPACE_ID,
        model=_MODEL_ID,
    )
    assert embedder.gradient_access_token == _GRADIENT_SECRET
    assert embedder.gradient_api_url == _GRADIENT_BASE_URL
    assert embedder.gradient_workspace_id == _GRADIENT_WORKSPACE_ID
    assert embedder.model == _MODEL_ID

    response = embedder.embed_documents(_DOCUMENTS * 1024)
    want = [
        [1.0, 0.0, 0.0],  # pizza
        [1.0, 0.0, 0.1],  # pizza  + long
        [0.0, 0.9, 0.0],  # doc
        [1.0, 0.0, 0.1],  # pizza + long
        [0.0, 0.9, 0.1],  # doc + long
    ] * 1024

    assert response == want


def test_gradient_wrong_setup(
    mocker: MockerFixture,
) -> None:
    mocker.patch("requests.post", side_effect=mocked_requests_post)

    with pytest.raises(Exception):
        GradientEmbeddings(
            gradient_api_url=_GRADIENT_BASE_URL,
            gradient_access_token="",  # empty
            gradient_workspace_id=_GRADIENT_WORKSPACE_ID,
            model=_MODEL_ID,
        )

    with pytest.raises(Exception):
        GradientEmbeddings(
            gradient_api_url=_GRADIENT_BASE_URL,
            gradient_access_token=_GRADIENT_SECRET,
            gradient_workspace_id="",  # empty
            model=_MODEL_ID,
        )

    with pytest.raises(Exception):
        GradientEmbeddings(
            gradient_api_url="-",  # empty
            gradient_access_token=_GRADIENT_SECRET,
            gradient_workspace_id=_GRADIENT_WORKSPACE_ID,
            model=_MODEL_ID,
        )
