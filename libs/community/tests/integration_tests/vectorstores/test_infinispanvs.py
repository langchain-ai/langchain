"""Test Infinispan functionality."""

import warnings
from typing import Any, List, Optional

import httpx
import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores.infinispanvs import InfinispanVS
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)

"""
cd tests/integration_tests/vectorstores/docker-compose
./infinispan.sh

Current Infinispan implementation relies on httpx: `pip install "httpx[http2]"`
if not installed. HTTP/2 is enable by default, if it's not
wanted use `pip install "httpx"`.
"""


def _infinispan_setup_noautoconf(**kwargs: Any) -> None:
    ispnvs = InfinispanVS(http2=_hasHttp2(), auto_config=False, **kwargs)
    ispnvs.cache_delete()
    ispnvs.schema_delete()
    proto = """
    /**
     * @Indexed
     */
    message vector {
    /**
     * @Vector(dimension=10)
     */
    repeated float vector = 1;
    optional string text = 2;
    optional string label = 3;
    optional int32 page = 4;
    }
    """
    ispnvs.schema_create(proto)
    ispnvs.cache_create()
    ispnvs.cache_index_clear()


def _infinispanvs_from_texts(
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
    clear_old: Optional[bool] = True,
    auto_config: Optional[bool] = False,
    **kwargs: Any,
) -> InfinispanVS:
    texts = [{"text": t} for t in fake_texts]
    if metadatas is None:
        metadatas = texts
    else:
        [m.update(t) for (m, t) in zip(metadatas, texts)]
    return InfinispanVS.from_texts(
        fake_texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        ids=ids,
        clear_old=clear_old,
        auto_config=auto_config,
        http2=_hasHttp2(),
        **kwargs,
    )


def _hasHttp2() -> bool:
    try:
        httpx.Client(http2=True)
        return True
    except Exception:
        return False


@pytest.mark.parametrize("autoconfig", [False, True])
@pytest.mark.parametrize(
    "conn_opts",
    [
        {},
        {
            "user": "user",
            "password": "password",
            "hosts": ["localhost:11232"],
            "schema": "http",
        },
        {
            "user": "user",
            "password": "password",
            "hosts": ["localhost:11242"],
            "schema": "https",
            "verify": False,
        },
    ],
)
class TestBasic:
    def test_infinispan(self, autoconfig: bool, conn_opts: dict) -> None:
        """Test end to end construction and search."""
        if not autoconfig:
            _infinispan_setup_noautoconf(**conn_opts)
        docsearch = _infinispanvs_from_texts(auto_config=autoconfig, **conn_opts)
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

    def test_infinispan_with_auth(self, autoconfig: bool, conn_opts: dict) -> None:
        """Test end to end construction and search."""
        if not autoconfig:
            _infinispan_setup_noautoconf(**conn_opts)
        docsearch = _infinispanvs_from_texts(auto_config=autoconfig, **conn_opts)
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

    def test_infinispan_with_metadata(self, autoconfig: bool, conn_opts: dict) -> None:
        """Test with metadata"""
        if not autoconfig:
            _infinispan_setup_noautoconf(**conn_opts)
        meta = []
        for _ in range(len(fake_texts)):
            meta.append({"label": "test"})
        docsearch = _infinispanvs_from_texts(
            metadatas=meta, auto_config=autoconfig, **conn_opts
        )
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={"label": "test"})]

    def test_infinispan_with_metadata_with_output_fields(
        self, autoconfig: bool, conn_opts: dict
    ) -> None:
        """Test with metadata"""
        if not autoconfig:
            _infinispan_setup_noautoconf(**conn_opts)
        metadatas = [
            {"page": i, "label": "label" + str(i)} for i in range(len(fake_texts))
        ]
        c = {"output_fields": ["label", "page", "text"]}
        docsearch = _infinispanvs_from_texts(
            metadatas=metadatas, configuration=c, auto_config=autoconfig, **conn_opts
        )
        output = docsearch.similarity_search("foo", k=1)
        assert output == [
            Document(page_content="foo", metadata={"label": "label0", "page": 0})
        ]

    def test_infinispanvs_with_id(self, autoconfig: bool, conn_opts: dict) -> None:
        """Test with ids"""
        ids = ["id_" + str(i) for i in range(len(fake_texts))]
        docsearch = _infinispanvs_from_texts(
            ids=ids, auto_config=autoconfig, **conn_opts
        )
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo")]

    def test_infinispan_with_score(self, autoconfig: bool, conn_opts: dict) -> None:
        """Test end to end construction and search with scores and IDs."""
        if not autoconfig:
            _infinispan_setup_noautoconf(**conn_opts)
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = _infinispanvs_from_texts(
            metadatas=metadatas, auto_config=autoconfig, **conn_opts
        )
        output = docsearch.similarity_search_with_score("foo", k=3)
        docs = [o[0] for o in output]
        scores = [o[1] for o in output]
        assert docs == [
            Document(page_content="foo", metadata={"page": 0}),
            Document(page_content="bar", metadata={"page": 1}),
            Document(page_content="baz", metadata={"page": 2}),
        ]
        assert scores[0] >= scores[1] >= scores[2]

    def test_infinispan_add_texts(self, autoconfig: bool, conn_opts: dict) -> None:
        """Test end to end construction and MRR search."""
        if not autoconfig:
            _infinispan_setup_noautoconf(**conn_opts)
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = _infinispanvs_from_texts(
            metadatas=metadatas, auto_config=autoconfig, **conn_opts
        )

        docsearch.add_texts(texts, metadatas)

        output = docsearch.similarity_search("foo", k=10)
        assert len(output) == 6

    def test_infinispan_no_clear_old(self, autoconfig: bool, conn_opts: dict) -> None:
        """Test end to end construction and MRR search."""
        if not autoconfig:
            _infinispan_setup_noautoconf(**conn_opts)
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = _infinispanvs_from_texts(
            metadatas=metadatas, auto_config=autoconfig, **conn_opts
        )
        del docsearch
        try:
            docsearch = _infinispanvs_from_texts(
                metadatas=metadatas,
                clear_old=False,
                auto_config=autoconfig,
                **conn_opts,
            )
        except AssertionError:
            if autoconfig:
                return
            else:
                raise
        output = docsearch.similarity_search("foo", k=10)
        assert len(output) == 6


class TestHttp2:
    def test_http2(self) -> None:
        try:
            httpx.Client(http2=True)
        except Exception:
            warnings.warn('pip install "httpx[http2]" if you need HTTP/2')
        pass
