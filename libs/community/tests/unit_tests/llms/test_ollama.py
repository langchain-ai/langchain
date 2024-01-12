import requests
from pytest import MonkeyPatch

from langchain_community.llms.ollama import Ollama


def mock_response_stream():
    mock_response = [b'{ "response": "Response chunk 1" }']

    class MockRaw:
        def read(self, chunk_size):
            try:
                return mock_response.pop()
            except IndexError:
                return None

    response = requests.Response()
    response.status_code = 200
    response.raw = MockRaw()
    return response


def test_pass_headers_if_provided(monkeypatch: MonkeyPatch) -> None:
    llm = Ollama(
        base_url="https://ollama-hostname:8000",
        model="foo",
        headers={
            "Authentication": "Bearer TEST-TOKEN-VALUE",
            "Referer": "https://application-host",
        },
        timeout=300,
    )

    def mockPost(url, headers, json, stream, timeout):
        assert url == "https://ollama-hostname:8000/api/generate/"
        assert headers == {
            "Content-Type": "application/json",
            "Authentication": "Bearer TEST-TOKEN-VALUE",
            "Referer": "https://application-host",
        }
        assert json is not None
        assert stream is True
        assert timeout == 300

        return mock_response_stream()

    monkeypatch.setattr(requests, "post", mockPost)

    llm("Test prompt")


def test_handle_if_headers_not_provided(monkeypatch: MonkeyPatch) -> None:
    llm = Ollama(base_url="https://ollama-hostname:8000", model="foo", timeout=300)

    def mockPost(url, headers, json, stream, timeout):
        assert url == "https://ollama-hostname:8000/api/generate/"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json is not None
        assert stream is True
        assert timeout == 300

        return mock_response_stream()

    monkeypatch.setattr(requests, "post", mockPost)

    llm("Test prompt")
