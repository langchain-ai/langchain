import json

import numpy as np
import requests
from pytest import MonkeyPatch

from langchain_community.embeddings import LlamafileEmbeddings


def mock_response() -> requests.Response:
    contents = json.dumps({"embedding": np.random.randn(512).tolist()})
    response = requests.Response()
    response.status_code = 200
    response._content = str.encode(contents)
    return response


def test_embed_documents(monkeypatch: MonkeyPatch) -> None:
    """
    Test basic functionality of the `embed_documents` method
    """
    embedder = LlamafileEmbeddings(
        base_url="http://llamafile-host:8080",
    )

    def mock_post(url, headers, json, timeout):  # type: ignore[no-untyped-def]
        assert url == "http://llamafile-host:8080/embedding"
        assert headers == {
            "Content-Type": "application/json",
        }
        # 'unknown' kwarg should be ignored
        assert json == {"content": "Test text"}
        # assert stream is False
        assert timeout is None
        return mock_response()

    monkeypatch.setattr(requests, "post", mock_post)
    out = embedder.embed_documents(["Test text", "Test text"])
    assert isinstance(out, list)
    assert len(out) == 2
    for vec in out:
        assert len(vec) == 512


def test_embed_query(monkeypatch: MonkeyPatch) -> None:
    """
    Test basic functionality of the `embed_query` method
    """
    embedder = LlamafileEmbeddings(
        base_url="http://llamafile-host:8080",
    )

    def mock_post(url, headers, json, timeout):  # type: ignore[no-untyped-def]
        assert url == "http://llamafile-host:8080/embedding"
        assert headers == {
            "Content-Type": "application/json",
        }
        # 'unknown' kwarg should be ignored
        assert json == {"content": "Test text"}
        # assert stream is False
        assert timeout is None
        return mock_response()

    monkeypatch.setattr(requests, "post", mock_post)
    out = embedder.embed_query("Test text")
    assert isinstance(out, list)
    assert len(out) == 512
