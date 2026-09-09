"""Regression tests for chains that destructure caller-owned input mappings."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from langchain_core.documents import Document
from requests import Response

from langchain_classic.chains.mapreduce import MapReduceChain
from langchain_classic.chains.openai_functions.openapi import SimpleRequestChain
from langchain_classic.chains.qa_with_sources.base import QAWithSourcesChain


class _TextSplitter:
    def split_text(self, text: str) -> list[str]:
        return [text]


class _CombineDocumentsChain:
    input_key = "input_documents"

    def run(self, inputs: dict[str, Any] | None = None, **_kwargs: Any) -> str:
        return "combined" if inputs is not None else "answer\nSOURCES: source"

    async def arun(self, **_kwargs: Any) -> str:
        return "answer\nSOURCES: source"


def test_map_reduce_call_does_not_mutate_inputs() -> None:
    chain = SimpleNamespace(
        input_key="input_text",
        output_key="output_text",
        text_splitter=_TextSplitter(),
        combine_documents_chain=_CombineDocumentsChain(),
    )
    inputs = {"input_text": "hello", "metadata": "kept"}
    original = deepcopy(inputs)

    result = MapReduceChain._call(chain, inputs)

    assert result == {"output_text": "combined"}
    assert inputs == original


def test_qa_with_sources_call_does_not_mutate_inputs() -> None:
    chain = SimpleNamespace(input_docs_key="docs")
    inputs = {"question": "question", "docs": [Document(page_content="context")]}
    original = deepcopy(inputs)

    docs = QAWithSourcesChain._get_docs(chain, inputs, run_manager=None)

    assert docs == inputs["docs"]
    assert inputs == original


async def test_qa_with_sources_acall_does_not_mutate_inputs() -> None:
    chain = SimpleNamespace(input_docs_key="docs")
    inputs = {"question": "question", "docs": [Document(page_content="context")]}
    original = deepcopy(inputs)

    docs = await QAWithSourcesChain._aget_docs(chain, inputs, run_manager=None)

    assert docs == inputs["docs"]
    assert inputs == original


def test_openapi_request_call_does_not_mutate_nested_input() -> None:
    response = Response()
    response.status_code = 200
    response._content = b'{"ok": true}'
    chain = SimpleRequestChain(request_method=lambda _name, _args: response)
    inputs = {
        "function": {
            "name": "get_item",
            "arguments": {"params": {"id": "1"}},
        }
    }
    original = deepcopy(inputs)

    chain._call(inputs)

    assert inputs == original
