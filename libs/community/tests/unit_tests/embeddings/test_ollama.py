import requests
from pytest import MonkeyPatch

from langchain_community.embeddings.ollama import OllamaEmbeddings


class MockResponse:
    status_code = 200

    def json(self) -> dict:
        return {"embedding": [1, 2, 3]}


def mock_response() -> MockResponse:
    return MockResponse()


def test_pass_headers_if_provided(monkeypatch: MonkeyPatch) -> None:
    embedder = OllamaEmbeddings(
        base_url="https://ollama-hostname:8000",
        model="foo",
        headers={
            "Authorization": "Bearer TEST-TOKEN-VALUE",
            "Referer": "https://application-host",
        },
    )

    def mock_post(url: str, headers: dict, json: str) -> MockResponse:
        assert url == "https://ollama-hostname:8000/api/embeddings"
        assert headers == {
            "Content-Type": "application/json",
            "Authorization": "Bearer TEST-TOKEN-VALUE",
            "Referer": "https://application-host",
        }
        assert json is not None

        return mock_response()

    monkeypatch.setattr(requests, "post", mock_post)

    embedder.embed_query("Test prompt")


def test_handle_if_headers_not_provided(monkeypatch: MonkeyPatch) -> None:
    embedder = OllamaEmbeddings(
        base_url="https://ollama-hostname:8000",
        model="foo",
    )

    def mock_post(url: str, headers: dict, json: str) -> MockResponse:
        assert url == "https://ollama-hostname:8000/api/embeddings"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json is not None

        return mock_response()

    monkeypatch.setattr(requests, "post", mock_post)

    embedder.embed_query("Test prompt")
