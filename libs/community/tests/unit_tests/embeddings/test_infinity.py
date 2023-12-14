from typing import Dict

from pytest_mock import MockerFixture

from langchain_community.embeddings import InfinityEmbeddings

_MODEL_ID = "BAAI/bge-small"
_INFINITY_BASE_URL = "https://localhost/api"
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
    assert url.startswith(_INFINITY_BASE_URL)
    assert "model" in json and _MODEL_ID in json["model"]
    assert json
    assert headers

    assert "input" in json and isinstance(json["input"], list)
    embeddings = []
    for inp in json["input"]:
        # verify correct ordering
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
        json_data={"data": embeddings},
        status_code=200,
    )


def test_infinity_emb_sync(
    mocker: MockerFixture,
) -> None:
    mocker.patch("requests.post", side_effect=mocked_requests_post)

    embedder = InfinityEmbeddings(model=_MODEL_ID, infinity_api_url=_INFINITY_BASE_URL)

    assert embedder.infinity_api_url == _INFINITY_BASE_URL
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


def test_infinity_large_batch_size(
    mocker: MockerFixture,
) -> None:
    mocker.patch("requests.post", side_effect=mocked_requests_post)

    embedder = InfinityEmbeddings(
        infinity_api_url=_INFINITY_BASE_URL,
        model=_MODEL_ID,
    )

    assert embedder.infinity_api_url == _INFINITY_BASE_URL
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
