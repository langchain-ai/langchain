"""Integration test for Embedchain."""

import os
from unittest.mock import patch

import pytest
from embedchain import Pipeline

from langchain.retrievers.embedchain import EmbedchainRetriever
from langchain.schema import Document

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
def test_embedchain_retriever(mock_add, mock_search) -> None:
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
