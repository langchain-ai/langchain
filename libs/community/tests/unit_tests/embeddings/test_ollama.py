import requests
from pytest import MonkeyPatch

from langchain_community.embeddings.ollama import OllamaEmbeddings


def mock_response():
    class MockResponse:
        status_code = 200

        def json(self):
            return {"embedding": [1, 2, 3]}

    return MockResponse()


def test_pass_headers_if_provided(monkeypatch: MonkeyPatch) -> None:
    embedder = OllamaEmbeddings(
        base_url="https://ollama-hostname:8000",
        model="foo",
        headers={
            "Authentication": "Bearer TEST-TOKEN-VALUE",
            "Referer": "https://application-host",
        },
    )

    def mock_post(url, headers, json):
        assert url == "https://ollama-hostname:8000/api/embeddings"
        assert headers == {
            "Content-Type": "application/json",
            "Authentication": "Bearer TEST-TOKEN-VALUE",
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

    def mock_post(url, headers, json):
        assert url == "https://ollama-hostname:8000/api/embeddings"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json is not None

        return mock_response()

    monkeypatch.setattr(requests, "post", mock_post)

    embedder.embed_query("Test prompt")
