from __future__ import annotations

import inspect
import uuid
from types import TracebackType
from typing import Any, Type

import aiohttp
import pytest
import requests_mock

from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

link_to_one_two = """
<div><a href="/one">link_to_one</a></div>
<div><a href="/two">link_to_two</a></div>
"""
link_to_three = '<div><a href="../three">link_to_three</a></div>'
no_links = "<p>no links<p>"

fake_url = f"https://{uuid.uuid4()}.com"
URL_TO_HTML = {
    fake_url: link_to_one_two,
    f"{fake_url}/one": link_to_three,
    f"{fake_url}/two": link_to_three,
    f"{fake_url}/three": no_links,
}


class MockGet:
    def __init__(self, url: str) -> None:
        self._text = URL_TO_HTML[url]
        self.headers: dict = {}

    async def text(self) -> str:
        return self._text

    async def __aexit__(
        self, exc_type: Type[BaseException], exc: BaseException, tb: TracebackType
    ) -> None:
        pass

    async def __aenter__(self) -> MockGet:
        return self


@pytest.mark.parametrize(("max_depth", "expected_docs"), [(1, 1), (2, 3), (3, 4)])
@pytest.mark.parametrize("use_async", [False, True])
def test_lazy_load(
    mocker: Any, max_depth: int, expected_docs: int, use_async: bool
) -> None:
    loader = RecursiveUrlLoader(fake_url, max_depth=max_depth, use_async=use_async)
    if use_async:
        mocker.patch.object(aiohttp.ClientSession, "get", new=MockGet)
        docs = list(loader.lazy_load())
    else:
        with requests_mock.Mocker() as m:
            for url, html in URL_TO_HTML.items():
                m.get(url, text=html)
            docs = list(loader.lazy_load())
    assert len(docs) == expected_docs


@pytest.mark.parametrize(("max_depth", "expected_docs"), [(1, 1), (2, 3), (3, 4)])
@pytest.mark.parametrize("use_async", [False, True])
async def test_alazy_load(
    mocker: Any, max_depth: int, expected_docs: int, use_async: bool
) -> None:
    loader = RecursiveUrlLoader(fake_url, max_depth=max_depth, use_async=use_async)
    if use_async:
        mocker.patch.object(aiohttp.ClientSession, "get", new=MockGet)
        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)
    else:
        with requests_mock.Mocker() as m:
            for url, html in URL_TO_HTML.items():
                m.get(url, text=html)
            docs = []
            async for doc in loader.alazy_load():
                docs.append(doc)

    assert len(docs) == expected_docs


def test_init_args_documented() -> None:
    cls_docstring = RecursiveUrlLoader.__doc__ or ""
    init_docstring = RecursiveUrlLoader.__init__.__doc__ or ""
    all_docstring = cls_docstring + init_docstring
    init_args = list(inspect.signature(RecursiveUrlLoader.__init__).parameters)
    undocumented = [arg for arg in init_args[1:] if f"{arg}:" not in all_docstring]
    assert not undocumented


@pytest.mark.parametrize("method", ["load", "aload", "lazy_load", "alazy_load"])
def test_no_runtime_args(method: str) -> None:
    method_attr = getattr(RecursiveUrlLoader, method)
    args = list(inspect.signature(method_attr).parameters)
    assert args == ["self"]
