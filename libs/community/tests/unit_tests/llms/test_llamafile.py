import json
from collections import deque

import pytest
import requests
from pytest import MonkeyPatch

from langchain_community.llms.llamafile import Llamafile


def mock_response():
    contents = {"content": "the quick brown fox"}
    contents = json.dumps(contents)
    response = requests.Response()
    response.status_code = 200
    response._content = str.encode(contents)
    return response


def mock_response_stream():  # type: ignore[no-untyped-def]
    mock_response = deque(
        [
            b'data: {"content":"the","multimodal":false,"slot_id":0,"stop":false}\n\n',
            b'data: {"content":" quick","multimodal":false,"slot_id":0,"stop":false}\n\n',
        ]
    )

    class MockRaw:
        def read(self, chunk_size):  # type: ignore[no-untyped-def]
            try:
                return mock_response.popleft()
            except IndexError:
                return None

    response = requests.Response()
    response.status_code = 200
    response.raw = MockRaw()
    return response


def test_call(monkeypatch: MonkeyPatch):
    """
    Test basic functionality of the `invoke` method
    """
    llm = Llamafile(
        base_url="http://llamafile-host:8080",
    )

    def mock_post(url, headers, json, stream, timeout):
        assert url == "http://llamafile-host:8080/completion"
        assert headers == {
            "Content-Type": "application/json",
        }
        # 'unknown' kwarg should be ignored
        assert json == {
            "prompt": "Test prompt",
            "temperature": 0.8,
            "seed": -1,
        }
        assert stream is False
        assert timeout is None
        return mock_response()

    monkeypatch.setattr(requests, "post", mock_post)
    out = llm.invoke("Test prompt")
    assert out == "the quick brown fox"


def test_call_with_kwargs(monkeypatch: MonkeyPatch):
    """
    Test kwargs passed to `invoke` override the default values and are passed
    to the endpoint correctly. Also test that any 'unknown' kwargs that are not
    present in the LLM class attrs are ignored.
    """
    llm = Llamafile(
        base_url="http://llamafile-host:8080",
    )

    def mock_post(url, headers, json, stream, timeout):
        assert url == "http://llamafile-host:8080/completion"
        assert headers == {
            "Content-Type": "application/json",
        }
        # 'unknown' kwarg should be ignored
        assert json == {
            "prompt": "Test prompt",
            "temperature": 0.8,
            "seed": 0,
        }
        assert stream is False
        assert timeout is None
        return mock_response()

    monkeypatch.setattr(requests, "post", mock_post)
    out = llm.invoke(
        "Test prompt",
        unknown="unknown option",  # should be ignored
        seed=0,  # should override the default
    )
    assert out == "the quick brown fox"


def test_call_raises_exception_on_missing_server(monkeypatch: MonkeyPatch):
    """
    Test that the LLM raises a ConnectionError when no llamafile server is
    listening at the base_url.
    """
    llm = Llamafile(
        # invalid url, nothing should actually be running here
        base_url="http://llamafile-host:8080",
    )
    with pytest.raises(requests.exceptions.ConnectionError):
        llm.invoke("Test prompt")


def test_streaming(monkeypatch: MonkeyPatch):
    """
    Test basic functionality of `invoke` with streaming enabled.
    """
    llm = Llamafile(
        base_url="http://llamafile-hostname:8080",
        streaming=True,
    )

    def mock_post(url, headers, json, stream, timeout):  # type: ignore[no-untyped-def]
        assert url == "http://llamafile-hostname:8080/completion"
        assert headers == {
            "Content-Type": "application/json",
        }
        # 'unknown' kwarg should be ignored
        assert "unknown" not in json
        assert json == {
            "prompt": "Test prompt",
            "temperature": 0.8,
            "seed": -1,
            "stream": True,
        }
        assert stream is True
        assert timeout is None

        return mock_response_stream()

    monkeypatch.setattr(requests, "post", mock_post)
    out = llm.invoke("Test prompt")
    assert out == "the quick"
