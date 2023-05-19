import importlib
import os
import uuid
from typing import List

import pytest

from langchain.docstore.document import Document
from langchain.vectorstores.vectara import Vectara


def test_vectara_add_documents() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    customer_id = os.environ['VECTARA_CUSTOMER_ID']
    corpus_id = os.environ['VECTARA_CORPUS_ID']
    api_key = os.environ['VECTARA_API_KEY']

    docsearch: Vectara = Vectara.from_texts(texts, customer_id, corpus_id, api_key)

    new_texts = ["foobar", "foobaz"]
    docsearch.add_documents([Document(page_content=content) for content in new_texts])
    output = docsearch.similarity_search("foobar", k=1)
    print(output)
    assert output == [Document(page_content="foobar")] or output == [
        Document(page_content="foo")
    ]
