import os
import uuid

import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores.huaweicloud_vector_search import CSSVectorStore
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)


class TestCSSVectorStore:
    @pytest.fixture(scope="class", autouse=True)
    def css_url(self) -> str:
        return os.environ.get("CSS_URL", "http://localhost:9200")

    @pytest.fixture(scope="function")
    def index_name(self) -> str:
        return f"test_{uuid.uuid4().hex}"

    def test_knn_similarity_search(self, css_url: str, index_name: str):
        store = CSSVectorStore.from_texts(
            fake_texts,
            FakeEmbeddings(),
            css_url=css_url,
            index_name=index_name,
            indexing=False,
        )
        output = store.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

    def test_ann_similarity_search(self, css_url: str, index_name: str):
        store = CSSVectorStore.from_texts(
            fake_texts,
            FakeEmbeddings(),
            css_url=css_url,
            index_name=index_name,
            indexing=True,
        )
        output = store.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]
