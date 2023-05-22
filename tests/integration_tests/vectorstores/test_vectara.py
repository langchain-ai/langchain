import os
from typing import List

import pytest

from langchain.docstore.document import Document
from langchain.vectorstores.vectara import Vectara
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_vectara_add_documents() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    customer_id = int(os.environ["VECTARA_CUSTOMER_ID"])
    corpus_id = int(os.environ["VECTARA_CORPUS_ID"])
    api_key = os.environ["VECTARA_API_KEY"]

    docsearch: Vectara = Vectara.from_texts(
        texts,
        embedding=FakeEmbeddings(),
        metadatas=None,
        customer_id=customer_id,
        corpus_id=corpus_id,
        api_key=api_key,
    )

    new_texts = ["foobar", "foobaz"]
    docsearch.add_documents([Document(page_content=content) for content in new_texts])
    output = docsearch.similarity_search("foobar", k=1)
    assert output == [Document(page_content="foobar")] or output == [
        Document(page_content="foo")
    ]
