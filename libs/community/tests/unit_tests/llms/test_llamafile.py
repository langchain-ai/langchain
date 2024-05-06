import json
from collections import deque
from typing import Any, Dict

import pytest
import requests
from pytest import MonkeyPatch

from langchain_community.llms.llamafile import Llamafile


def default_generation_params() -> Dict[str, Any]:
    return {
        "temperature": 0.8,
        "seed": -1,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        "n_predict": -1,
        "n_keep": 0,
        "tfs_z": 1.0,
        "typical_p": 1.0,
        "repeat_penalty": 1.1,
        "repeat_last_n": 64,
        "penalize_nl": True,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "mirostat": 0,
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1,
    }


def mock_response() -> requests.Response:
    contents = json.dumps({"content": "the quick brown fox"})
    response = requests.Response()
    response.status_code = 200
    response._content = str.encode(contents)
    return response


def mock_response_stream():  # type: ignore[no-untyped-def]
    mock_response = deque(
        [
            b'data: {"content":"the","multimodal":false,"slot_id":0,"stop":false}\n\n',  # noqa
            b'data: {"content":" quick","multimodal":false,"slot_id":0,"stop":false}\n\n',  # noqa
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


def test_call(monkeypatch: MonkeyPatch) -> None:
    """
    Test basic functionality of the `invoke` method
    """
    llm = Llamafile(
        base_url="http://llamafile-host:8080",
    )

    def mock_post(url, headers, json, stream, timeout):  # type: ignore[no-untyped-def]
        assert url == "http://llamafile-host:8080/completion"
        assert headers == {
            "Content-Type": "application/json",
        }
        # 'unknown' kwarg should be ignored
        assert json == {"prompt": "Test prompt", **default_generation_params()}
        assert stream is False
        assert timeout is None
        return mock_response()

    monkeypatch.setattr(requests, "post", mock_post)
    out = llm.invoke("Test prompt")
    assert out == "the quick brown fox"


def test_call_with_kwargs(monkeypatch: MonkeyPatch) -> None:
    """
    Test kwargs passed to `invoke` override the default values and are passed
    to the endpoint correctly. Also test that any 'unknown' kwargs that are not
    present in the LLM class attrs are ignored.
    """
    llm = Llamafile(
        base_url="http://llamafile-host:8080",
    )

    def mock_post(url, headers, json, stream, timeout):  # type: ignore[no-untyped-def]
        assert url == "http://llamafile-host:8080/completion"
        assert headers == {
            "Content-Type": "application/json",
        }
        # 'unknown' kwarg should be ignored
        expected = {"prompt": "Test prompt", **default_generation_params()}
        expected["seed"] = 0
        assert json == expected
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


def test_call_raises_exception_on_missing_server(monkeypatch: MonkeyPatch) -> None:
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


def test_streaming(monkeypatch: MonkeyPatch) -> None:
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
        expected = {"prompt": "Test prompt", **default_generation_params()}
        expected["stream"] = True
        assert json == expected
        assert stream is True
        assert timeout is None

        return mock_response_stream()

    monkeypatch.setattr(requests, "post", mock_post)
    out = llm.invoke("Test prompt")
    assert out == "the quick"
