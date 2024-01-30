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

    def mock_post(url, headers, json, stream, timeout):
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

    monkeypatch.setattr(requests, "post", mock_post)

    llm("Test prompt")


def test_handle_if_headers_not_provided(monkeypatch: MonkeyPatch) -> None:
    llm = Ollama(base_url="https://ollama-hostname:8000", model="foo", timeout=300)

    def mock_post(url, headers, json, stream, timeout):
        assert url == "https://ollama-hostname:8000/api/generate/"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json is not None
        assert stream is True
        assert timeout == 300

        return mock_response_stream()

    monkeypatch.setattr(requests, "post", mock_post)

    llm("Test prompt")


def test_handle_kwargs_top_level_parameters(monkeypatch: MonkeyPatch) -> None:
    """Test that top level params are sent to the endpoint as top level params"""
    llm = Ollama(base_url="https://ollama-hostname:8000", model="foo", timeout=300)

    def mock_post(url, headers, json, stream, timeout):
        assert url == "https://ollama-hostname:8000/api/generate/"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json == {
            "format": None,
            "images": None,
            "model": "test-model",
            "options": {
                "mirostat": None,
                "mirostat_eta": None,
                "mirostat_tau": None,
                "num_ctx": None,
                "num_gpu": None,
                "num_thread": None,
                "num_predict": None,
                "repeat_last_n": None,
                "repeat_penalty": None,
                "stop": [],
                "temperature": None,
                "tfs_z": None,
                "top_k": None,
                "top_p": None,
            },
            "prompt": "Test prompt",
            "system": "Test system prompt",
            "template": None,
        }
        assert stream is True
        assert timeout == 300

        return mock_response_stream()

    monkeypatch.setattr(requests, "post", mock_post)

    llm("Test prompt", model="test-model", system="Test system prompt")


def test_handle_kwargs_with_unknown_param(monkeypatch: MonkeyPatch) -> None:
    """
    Test that params that are not top level params will be sent to the endpoint
    as options
    """
    llm = Ollama(base_url="https://ollama-hostname:8000", model="foo", timeout=300)

    def mock_post(url, headers, json, stream, timeout):
        assert url == "https://ollama-hostname:8000/api/generate/"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json == {
            "format": None,
            "images": None,
            "model": "foo",
            "options": {
                "mirostat": None,
                "mirostat_eta": None,
                "mirostat_tau": None,
                "num_ctx": None,
                "num_gpu": None,
                "num_thread": None,
                "num_predict": None,
                "repeat_last_n": None,
                "repeat_penalty": None,
                "stop": [],
                "temperature": 0.8,
                "tfs_z": None,
                "top_k": None,
                "top_p": None,
                "unknown": "Unknown parameter value",
            },
            "prompt": "Test prompt",
            "system": None,
            "template": None,
        }
        assert stream is True
        assert timeout == 300

        return mock_response_stream()

    monkeypatch.setattr(requests, "post", mock_post)

    llm("Test prompt", unknown="Unknown parameter value", temperature=0.8)


def test_handle_kwargs_with_options(monkeypatch: MonkeyPatch) -> None:
    """
    Test that if options provided it will be sent to the endpoint as options,
    ignoring other params that are not top level params.
    """
    llm = Ollama(base_url="https://ollama-hostname:8000", model="foo", timeout=300)

    def mock_post(url, headers, json, stream, timeout):
        assert url == "https://ollama-hostname:8000/api/generate/"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json == {
            "format": None,
            "images": None,
            "model": "test-another-model",
            "options": {"unknown_option": "Unknown option value"},
            "prompt": "Test prompt",
            "system": None,
            "template": None,
        }
        assert stream is True
        assert timeout == 300

        return mock_response_stream()

    monkeypatch.setattr(requests, "post", mock_post)

    llm(
        "Test prompt",
        model="test-another-model",
        options={"unknown_option": "Unknown option value"},
        unknown="Unknown parameter value",
        temperature=0.8,
    )
