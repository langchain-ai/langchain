"""Integration test for Embedchain."""

import os
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from langchain_community.retrievers.embedchain import EmbedchainRetriever

try:
    from embedchain import Pipeline
except ImportError:
    pytest.skip("Requires embedchain", allow_module_level=True)

os.environ["OPENAI_API_KEY"] = "sk-xxxx"

context_value = [
    {
        "context": "this document is about John",
        "source": "source#1",
        "document_id": 123,
    },
]


@pytest.mark.requires("embedchain")
@patch.object(Pipeline, "search", return_value=context_value)
@patch.object(Pipeline, "add", return_value=123)
def test_embedchain_retriever(mock_add: Any, mock_search: Any) -> None:
    retriever = EmbedchainRetriever.create()
    texts = [
        "This document is about John",
    ]
    for text in texts:
        retriever.add_texts(text)
    docs = retriever.get_relevant_documents("doc about john")
    assert len(docs) == 1
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata
        assert len(list(doc.metadata.items())) > 0
