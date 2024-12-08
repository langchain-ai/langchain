"""Test Lindorm Search Store."""

import asyncio
import copy
import logging
import os
import timeit
import uuid
from functools import partial
from importlib import util
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.lindorm_embedding import LindormAIEmbeddings
from langchain_community.vectorstores.lindorm_vector_search import LindormVectorStore

IMPORT_OPENSEARCH_PY_ERROR = (
    "Could not import OpenSearch. Please install it with `pip install opensearch-py`."
)


def _get_opensearch_scan() -> Any:
    if util.find_spec("opensearchpy.helpers"):
        from opensearchpy.helpers import scan

        return scan  # 返回 bulk 函数的引用
    else:
        raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)


class Config:
    AI_LLM_ENDPOINT = os.environ.get("AI_LLM_ENDPOINT", "<LLM_ENDPOINT>")
    AI_EMB_ENDPOINT = os.environ.get("AI_EMB_ENDPOINT", "<EMB_ENDPOINT>")
    AI_USERNAME = os.environ.get("AI_USERNAME", "root")
    AI_PWD = os.environ.get("AI_PWD", "<PASSWORD>")

    AI_DEFAULT_RERANK_MODEL = "rerank_bge_v2_m3"
    AI_DEFAULT_EMBEDDING_MODEL = "bge_m3_model"
    SEARCH_ENDPOINT = os.environ.get("SEARCH_ENDPOINT", "SEARCH_ENDPOINT")
    SEARCH_USERNAME = os.environ.get("SEARCH_USERNAME", "root")
    SEARCH_PWD = os.environ.get("SEARCH_PWD", "<PASSWORD>")


logger = logging.getLogger(__name__)


def get_default_embedding() -> Any:
    embedding = LindormAIEmbeddings(
        endpoint=Config.AI_LLM_ENDPOINT,
        username=Config.AI_USERNAME,
        password=Config.AI_PWD,
        model_name=Config.AI_DEFAULT_EMBEDDING_MODEL,
        client=None,
    )
    return embedding


BUILD_INDEX_PARAMS = {
    "lindorm_search_url": Config.SEARCH_ENDPOINT,
    # default params
    "embedding": get_default_embedding(),
    "http_auth": (Config.SEARCH_USERNAME, Config.SEARCH_PWD),
    "use_ssl": False,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "bulk_size": 500,
    "timeout": 60,
    "max_retries": 3,
    "retry_on_timeout": True,
    "embed_thread_num": 2,
    "write_thread_num": 5,
    "pool_maxsize": 20,
    "engine": "lvector",
    "space_type": "l2",
}

BUILD_INDEX = True


def test_build_index() -> Any:
    scan = _get_opensearch_scan()
    BUILD_INDEX_PARAMS["index_name"] = "tl_non_route_index"
    BUILD_INDEX_PARAMS["method_name"] = "hnsw"

    if BUILD_INDEX:

        def get_default_docs() -> Any:
            loader = TextLoader("./threekingdoms.txt")
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            for i, doc in enumerate(docs):
                doc.metadata = {"rating": i}

            return docs

        BUILD_INDEX_PARAMS["documents"] = get_default_docs()
        BUILD_INDEX_PARAMS["ids"] = [
            str(doc.metadata["rating"]) for doc in BUILD_INDEX_PARAMS["documents"]
        ]
        BUILD_INDEX_PARAMS["overwrite"] = False
        docsearch = LindormVectorStore.from_documents(**BUILD_INDEX_PARAMS)
    else:
        docsearch = LindormVectorStore(**BUILD_INDEX_PARAMS)

    ratings = []
    for hit in scan(docsearch.client, index=docsearch.index_name):
        ratings.append(hit["_source"]["metadata"]["rating"])

    logger.info("ratings: {}".format(ratings))
    for i in range(len(ratings)):
        assert i in ratings

    return docsearch


def test_build_route_index() -> Any:
    scan = _get_opensearch_scan()
    ROUTE_BUILD_INDEX_PARAMS = copy.deepcopy(BUILD_INDEX_PARAMS)
    ROUTE_BUILD_INDEX_PARAMS["index_name"] = "tl_route_index"
    ROUTE_BUILD_INDEX_PARAMS["routing_field"] = "split_setting"
    ROUTE_BUILD_INDEX_PARAMS["method_name"] = "ivfpq"
    ROUTE_BUILD_INDEX_PARAMS["nlist"] = 32
    if BUILD_INDEX:

        def get_default_docs() -> object:
            loader = TextLoader("./threekingdoms.txt")
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)

            docs = [copy.deepcopy(doc) for doc in docs for _ in range(10)]
            for i, doc in enumerate(docs):
                doc.metadata = {"rating": i, "split_setting": str(i % 5)}
            return docs

        ROUTE_BUILD_INDEX_PARAMS["documents"] = get_default_docs()
        ROUTE_BUILD_INDEX_PARAMS["ids"] = [
            str(doc.metadata["rating"]) for doc in ROUTE_BUILD_INDEX_PARAMS["documents"]
        ]
        ROUTE_BUILD_INDEX_PARAMS["overwrite"] = False

        logger.info(f"total doc: {len(ROUTE_BUILD_INDEX_PARAMS['documents'])}")
        # docsearch = LindormSearchStore(**ROUTE_BUILD_INDEX_PARAMS)
        # docsearch.delete_index(ROUTE_BUILD_INDEX_PARAMS["index_name"])
        # docsearch.from_documents(**ROUTE_BUILD_INDEX_PARAMS)

        docsearch = LindormVectorStore.from_documents(**ROUTE_BUILD_INDEX_PARAMS)
    else:
        docsearch = LindormVectorStore(**ROUTE_BUILD_INDEX_PARAMS)

    ratings = []
    for hit in scan(docsearch.client, index=docsearch.index_name):
        ratings.append(hit["_source"]["metadata"])
    SEARCH_INDEX_PARAMS = {
        "_source": True,
        "hybrid": True,
        "rrf_rank_constant": "60",
        "nprobe": "32",
        "reorder_factor": "2",
        "routing": "0",  # "0" or "1"
    }

    # knn
    query = "董卓"
    k = 20
    docs = docsearch.similarity_search(query=query, k=k, **SEARCH_INDEX_PARAMS)
    assert len(docs) > 0
    assert docs[0].id is not None
    logger.info(f"route ann {docs[0:1]}")

    # get_by_ids
    id_docs = docsearch.get_by_ids([doc.id for doc in docs if doc.id is not None])
    assert len(id_docs) == len(docs)

    return docsearch


def test_inherit_codebook(init_by_from: bool) -> Any:
    # init texts
    def get_default_docs() -> Any:
        loader = TextLoader("./threekingdoms.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        docs = [copy.deepcopy(doc) for doc in docs for _ in range(10)]
        for i, doc in enumerate(docs):
            doc.metadata = {"rating": i, "split_setting": str(i % 5)}
        return docs

    documents = get_default_docs()
    logger.info(f"total doc: {len(documents)}")  # 1100

    # add texts
    ROUTE_INHERIT_INDEX_PARAMS = copy.deepcopy(BUILD_INDEX_PARAMS)
    ROUTE_INHERIT_INDEX_PARAMS["index_name"] = (
        f"new_route_index_inherit_{str(uuid.uuid4())}_{str(init_by_from)}"
    )
    ROUTE_INHERIT_INDEX_PARAMS["parent_index"] = "tl_route_index"
    ROUTE_INHERIT_INDEX_PARAMS["routing_field"] = "split_setting"
    ROUTE_INHERIT_INDEX_PARAMS["method_name"] = "ivfpq"
    ROUTE_INHERIT_INDEX_PARAMS["overwrite"] = True
    ROUTE_INHERIT_INDEX_PARAMS["nlist"] = 32

    if init_by_from:
        ROUTE_INHERIT_INDEX_PARAMS["documents"] = documents
        ROUTE_INHERIT_INDEX_PARAMS["ids"] = [
            str(doc.metadata["rating"]) for doc in documents
        ]
        docsearch = LindormVectorStore.from_documents(**ROUTE_INHERIT_INDEX_PARAMS)
    else:
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        ids = [str(d.metadata["rating"]) for d in documents]
        docsearch = LindormVectorStore(**ROUTE_INHERIT_INDEX_PARAMS)
        docsearch.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    # search texts
    SEARCH_INDEX_PARAMS = {
        "_source": True,
        "search_type": "approximate_search",
        "hybrid": False,
        "rrf_rank_constant": "60",
        "nprobe": "32",
        "reorder_factor": "2",
        "routing": "0",  # "0" or "1"
    }

    query = "董卓"
    k = 20
    docs = docsearch.similarity_search(query=query, k=k, **SEARCH_INDEX_PARAMS)
    # knn
    assert len(docs) > 0
    assert docs[0].id is not None
    logger.info(f"route inherit ann:{docs[0:1]}")

    # get_by_ids
    id_docs = docsearch.get_by_ids([doc.id for doc in docs if doc.id is not None])
    assert len(id_docs) == len(docs)
    return docsearch


def test_appx_knn_search(docsearch: Any) -> None:
    SEARCH_INDEX_PARAMS = {
        "query": "董卓",
        "k": 10,
    }
    docs = docsearch.similarity_search(**SEARCH_INDEX_PARAMS)
    assert len(docs) > 0
    assert docs[0].id is not None
    logger.info(f"ann: {docs[0:1]}")

    id_docs = docsearch.get_by_ids([doc.id for doc in docs])
    assert len(id_docs) == len(docs)


def test_multi_query_hybrid_rrf_search(
    docsearch: Any, is_sync: bool = True, times: int = 10
) -> None:
    routing = str(2)
    SEARCH_INDEX_PARAMS = {
        "multi_query_rrf": [
            {
                "routing": routing,
                "query": "董卓",
                "_source": {"excludes": ["vector_field"]},
                "k": 10,
                "hybrid": True,
                "search_type": "approximate_search",
            },
            {
                "routing": routing,
                "query": "曹操",
                "_source": {"excludes": ["vector_field"]},
                "k": 10,
                "hybrid": True,
                "search_type": "approximate_search",
            },
            {
                "routing": routing,
                "query": "董卓和曹操",
                "_source": {"excludes": ["vector_field"]},
                "k": 10,
                "hybrid": True,
                "search_type": "approximate_search",
            },
        ]
    }

    def sync_func() -> Any:
        for i in range(times):
            docs = docsearch.similarity_search(**copy.deepcopy(SEARCH_INDEX_PARAMS))
            assert docs[0].id is not None
            docs = docsearch.similarity_search_with_score(
                **copy.deepcopy(SEARCH_INDEX_PARAMS)
            )
            assert docs[0][0].id is not None

    async def async_func() -> Any:
        tasks = list()
        for i in range(times):
            tasks.append(
                docsearch.asimilarity_search(**copy.deepcopy(SEARCH_INDEX_PARAMS))
            )
            tasks.append(
                docsearch.asimilarity_search_with_score(
                    **copy.deepcopy(SEARCH_INDEX_PARAMS)
                )
            )
        return await asyncio.gather(*tasks)

    if is_sync:
        sync_func()
    else:
        results = asyncio.run(async_func())
        assert len(results) == times * 2
        for i in range(times):
            assert results[i * 2][0].id is not None
            assert results[i * 2 + 1][0][0].id is not None


def test_multi_query_hybrid_rrf_search_different_query(docsearch: Any) -> None:
    routing = str(2)
    SEARCH_INDEX_PARAMS = {
        "multi_query_rrf": [
            # vector
            {
                "routing": routing,
                "query": "董卓",
                "_source": True,
                "k": 10,
                "hybrid": False,
                "rrf_rank_constant": "60",
                "search_type": "approximate_search",
            },
            # hybrid
            {
                "routing": routing,
                "query": "董卓",
                "_source": True,
                "k": 20,
                "hybrid": True,
                "rrf_rank_constant": "60",
                "search_type": "approximate_search",
            },
            # fulltext
            {
                "routing": routing,
                "query": "董卓",
                "_source": True,
                "k": 30,
                "hybrid": True,
                "rrf_rank_constant": "60",
                "search_type": "text_search",
            },
        ]
    }

    docs = docsearch.similarity_search(**copy.deepcopy(SEARCH_INDEX_PARAMS))
    assert docs[0].id is not None
    logger.info(f"multi_search_hybrid_rrf: {docs[0:1]}")

    docs = docsearch.similarity_search_with_score(**copy.deepcopy(SEARCH_INDEX_PARAMS))
    logger.info(f"multi_search_hybrid_rrf_withscore: {docs}")


def test_delete(docsearch: Any) -> None:
    routing = str(2)
    SEARCH_INDEX_PARAMS = {
        "routing": routing,
        "query": "董卓",
        "_source": True,
        "k": 10,
        "hybrid": True,
        "rrf_rank_constant": "60",
    }

    docs = docsearch.similarity_search(**copy.deepcopy(SEARCH_INDEX_PARAMS))
    assert docs[0].id is not None
    logger.info(f"hybrid_rrf: {docs[0:1]}")

    # delete by ids
    def delete_ids() -> Any:
        docsearch.refresh()
        assert len(docsearch.get_by_ids([doc.id for doc in docs])) == len(docs)
        rsp = docsearch.delete([doc.id for doc in docs])
        docsearch.refresh()
        assert len(docsearch.get_by_ids([doc.id for doc in docs])) == 0
        return rsp

    def delete_by_query(body: dict, params: dict) -> Any:
        return docsearch.delete_by_query(body=body, params=params)

    # delete index
    def delete_index() -> Any:
        return docsearch.delete_index(index_name=docsearch.index_name)

    logger.info(f"delete ids: {delete_ids()}")

    # first delete routing docs
    assert len(docsearch.similarity_search(**copy.deepcopy(SEARCH_INDEX_PARAMS))) > 0
    params = {"routing": routing, "wait_for_completion": "true"}
    body = {"query": {"range": {"metadata.rating": {"gte": 0, "lt": 5000}}}}
    logger.info(
        "delete by query with routing:{}".format(
            delete_by_query(body=body, params=params)
        )
    )
    docsearch.refresh()
    assert len(docsearch.similarity_search(**copy.deepcopy(SEARCH_INDEX_PARAMS))) == 0

    # second delete routing docs
    SEARCH_INDEX_PARAMS["routing"] = str(4)
    assert len(docsearch.similarity_search(**copy.deepcopy(SEARCH_INDEX_PARAMS))) > 0
    params = {"routing": str(4), "wait_for_completion": "true"}
    body2 = {"query": {"bool": {"must_not": {"term": {"age": "10"}}}}}
    logger.info(
        "delete by query with routing2:{}".format(
            delete_by_query(body=body2, params=params)
        )
    )
    docsearch.refresh()
    assert len(docsearch.similarity_search(**copy.deepcopy(SEARCH_INDEX_PARAMS))) <= 1

    # delete docs ignore routing
    SEARCH_INDEX_PARAMS["routing"] = str(3)
    assert len(docsearch.similarity_search(**copy.deepcopy(SEARCH_INDEX_PARAMS))) > 0
    params = {"wait_for_completion": "true"}
    body3 = {
        "query": {
            "bool": {"filter": {"range": {"metadata.rating": {"gte": 0, "lt": 5000}}}}
        }
    }
    logger.info(
        "delete by query without routing:{}".format(
            delete_by_query(body=body3, params=params)
        )
    )
    docsearch.refresh()
    assert len(docsearch.similarity_search(**copy.deepcopy(SEARCH_INDEX_PARAMS))) == 0

    delete_index()


def test_hybrid_rrf_search(docsearch: Any) -> Any:
    SEARCH_INDEX_PARAMS = {
        "query": "董卓",
        "_source": True,
        "k": 10,
        "hybrid": True,
        "rrf_rank_constant": "60",
    }
    docs = docsearch.similarity_search(**SEARCH_INDEX_PARAMS)
    assert docs[0].id is not None
    logger.info("hybrid_rrf:", docs[0:1])


def test_hybrid_rrf_search_withscore(docsearch: Any) -> Any:
    SEARCH_INDEX_PARAMS = {
        "query": "董卓",
        "_source": True,
        "k": 10,
        "hybrid": True,
        "rrf_rank_constant": "60",
    }
    docs = docsearch.similarity_search_with_score(**SEARCH_INDEX_PARAMS)
    logger.info(f"hybrid_rrf_withscore:{docs}")


def test_pre_filter(docsearch: Any) -> None:
    SEARCH_INDEX_PARAMS = {
        "query": "董卓",
        "k": 10,
        "filter": [{"range": {"metadata.rating": {"gte": 35}}}],
        "filter_type": "pre_filter",
    }
    docs = docsearch.similarity_search(**SEARCH_INDEX_PARAMS)
    assert docs[0].id is not None
    for doc in docs:
        assert doc.metadata["rating"] >= 35
    logger.info(f"pre_filter:{docs[0]}")


def test_post_filter(docsearch: Any) -> None:
    SEARCH_INDEX_PARAMS = {
        "query": "董卓",
        "k": 10,
        "filter": [{"range": {"metadata.rating": {"gt": 35, "lt": 50}}}],
        "filter_type": "post_filter",
    }
    docs = docsearch.similarity_search(**SEARCH_INDEX_PARAMS)
    assert docs[0].id is not None
    for doc in docs:
        assert doc.metadata["rating"] > 35
    logger.info(f"post_filter: {docs[0]}")


def test_multi_filter_with_rrf_off(docsearch: Any) -> None:
    SEARCH_INDEX_PARAMS = {
        "query": "董卓",
        "k": 10,
        "hybrid": False,
        "filter": [
            {"range": {"metadata.rating": {"gt": 20}}},
            {"range": {"metadata.rating": {"lt": 30}}},
        ],
        "filter_type": "pre_filter",  # or 'post_filter'
    }
    docs = docsearch.similarity_search(**SEARCH_INDEX_PARAMS)
    assert docs[0].id is not None
    for doc in docs:
        assert 20 < doc.metadata["rating"] < 30
    logger.info(f"multi_pre_filter: {docs[0]}")


def test_multi_filter_with_rrf_on(docsearch: Any) -> None:
    SEARCH_INDEX_PARAMS = {
        "query": "董卓",
        "k": 10,
        "hybrid": True,
        "rrf_rank_constant": "60",
        "filter": [
            {"range": {"metadata.rating": {"gt": 20}}},
            {"range": {"metadata.rating": {"lt": 30}}},
        ],
        "filter_type": "pre_filter",  # or 'post_filter'
    }
    docs = docsearch.similarity_search(**SEARCH_INDEX_PARAMS)
    assert docs[0].id is not None
    for doc in docs:
        assert 20 < doc.metadata["rating"] < 30
    logger.info(f"pre_filter_hybrid:{docs[0]}")


def test_search_with_min_score(docsearch: Any) -> None:
    SEARCH_INDEX_PARAMS = {"query": "董卓", "k": 10, "min_score": "0.5"}
    docs_with_scores = docsearch.similarity_search_with_score(**SEARCH_INDEX_PARAMS)

    for doc, score in docs_with_scores:
        assert score >= 0.5
    logger.info(f"ann_min_score:{docs_with_scores[0:1]}")


def test_text_search_with_must_clause(docsearch: Any) -> None:
    SEARCH_INDEX_PARAMS = {
        "search_type": "text_search",
        "query": "董卓",
        "k": 10,
        "must": [{"match": {"text": "曹操"}}],
    }
    docs = docsearch.similarity_search(**SEARCH_INDEX_PARAMS)
    assert docs[0].id is not None
    for doc in docs:
        assert "曹操" in doc.page_content
    logger.info(f"text_search_with_must:{docs[0:1]}")


def test_text_search_with_must_not_clause(docsearch: Any) -> None:
    SEARCH_INDEX_PARAMS = {
        "search_type": "text_search",
        "query": "刘备",
        "k": 10,
        "must_not": [{"match": {"text": "曹操"}}],
    }
    docs = docsearch.similarity_search(**SEARCH_INDEX_PARAMS)
    assert docs[0].id is not None
    for doc in docs:
        assert "曹操" not in doc.page_content
    logger.info(f"text_search_without_must: {docs[0:1]}")


def test_text_search_with_should_clause(docsearch: Any) -> None:
    SEARCH_INDEX_PARAMS = {
        "search_type": "text_search",
        "query": "刘备",
        "k": 10,
        "should": [{"match": {"text": "关羽"}}, {"match": {"text": "袁术"}}],
        "minimum_should_match": 1,
    }
    docs = docsearch.similarity_search(**SEARCH_INDEX_PARAMS)
    assert docs[0].id is not None
    for doc in docs:
        assert "关羽" in doc.page_content or "袁术" in doc.page_content
    logger.info(f"text_search_with_should:{docs[0:1]}")


def test_text_search_with_filter_clause(docsearch: Any) -> None:
    SEARCH_INDEX_PARAMS = {
        "search_type": "text_search",
        "query": "刘备",
        "k": 10,
        "filter": [
            {"range": {"metadata.rating": {"gt": 20}}},
            {"range": {"metadata.rating": {"lt": 30}}},
        ],
    }
    docs_with_scores = docsearch.similarity_search_with_score(**SEARCH_INDEX_PARAMS)
    for doc, score in docs_with_scores:
        assert 20 < doc.metadata["rating"] < 30
    logger.info(f"text_search_with_filter:{docs_with_scores[0:1]}")


def test_init() -> None:
    ldv = LindormVectorStore(
        lindorm_search_url=Config.SEARCH_ENDPOINT,
        index_name="test",
        embedding=get_default_embedding(),
        http_auth=(Config.SEARCH_USERNAME, Config.SEARCH_PWD),
        timeout=60,
        method_name="ivfpq",
        routing_field="split_setting",
        embed_thread_num=10,
        excludes_from_source=False,
        analyzer="ik_max_word",
    )
    # ldv.get_index_map()
    assert ldv.index_name == "test"


def test_route_hybrid_rrf_search(routesearch: Any) -> None:
    routing = str(1)
    SEARCH_INDEX_PARAMS = {
        "routing": routing,
        "query": "董卓",
        "_source": True,
        "k": 100,
        "hybrid": True,
        "rrf_rank_constant": "60",
        "nprobe": "32",
        "reorder_factor": "2",
    }
    docs = routesearch.similarity_search_with_score(**SEARCH_INDEX_PARAMS)
    find = False
    for doc in docs:
        assert doc[0].id is not None
        assert doc[0].metadata.get("split_setting") == routing
        if "search" in str(doc[2]) and "vector" in str(doc[2]):
            find = True
    assert find is True


def test_route_multi_filter_with_rrf_on(docsearch: Any) -> None:
    routing = str(2)
    SEARCH_INDEX_PARAMS = {
        "routing": routing,
        "query": "董卓",
        "k": 100,
        "hybrid": True,
        "rrf_rank_constant": "60",
        "filter": [
            {"range": {"metadata.rating": {"gt": 0}}},
            {"range": {"metadata.rating": {"lt": 300}}},
        ],
        "filter_type": "pre_filter",  # or 'post_filter'
        "nprobe": "32",
    }
    docs = docsearch.similarity_search_with_score(**SEARCH_INDEX_PARAMS)
    find = False
    for doc in docs:
        assert doc[0].id is not None
        assert doc[0].metadata.get("split_setting") == routing
        if "search" in str(doc[2]) and "vector" in str(doc[2]):
            find = True
        assert 0 < doc[0].metadata["rating"] < 300
    assert find


def test_rout_text_search_with_must_clause(docsearch: Any) -> None:
    routing = str(1)
    SEARCH_INDEX_PARAMS = {
        "routing": routing,
        "search_type": "text_search",
        "query": "董卓",
        "k": 100,
        "must": [{"match": {"text": "董卓"}}],
    }

    docs = docsearch.similarity_search(**SEARCH_INDEX_PARAMS)
    logger.info("docs size:", len(docs))

    assert docs[0].id is not None
    for doc in docs:
        assert doc.metadata.get("split_setting") == routing
        assert "董卓" in doc.page_content
    logger.info(
        f"rout_text_search_with_must_clause:{docs[0:1]}",
    )


async def async_test(routesearch: Any) -> None:
    routing = str(2)
    SEARCH_INDEX_PARAMS = {
        "routing": routing,
        "query": "董卓",
        "_source": True,
        "k": 10,
        "hybrid": True,
        "rrf_rank_constant": "60",
        "nprobe": "32",
        "reorder_factor": "2",
    }
    docs = await routesearch.asimilarity_search_with_score(**SEARCH_INDEX_PARAMS)
    from_vector = False
    from_search = False
    for doc in docs:
        assert doc[0].id is not None
        assert doc[0].metadata.get("split_setting") == routing
        if "search" in str(doc[2]):
            from_search = True
        elif "vector" in str(doc[2]):
            from_vector = True
    assert from_search and from_vector

    routing = str(1)
    id_docs = await routesearch.aget_by_ids([doc[0].id for doc in docs])
    assert len(id_docs) == len(docs)

    params = {"routing": routing, "wait_for_completion": "true"}
    body = {"query": {"range": {"metadata.rating": {"gte": 0, "lt": 1}}}}
    del_by_query_result = await routesearch.adelete_by_query(body, params)
    assert del_by_query_result["deleted"] >= 0

    add_docs = await routesearch.aadd_texts(
        texts=["foo", "bar", "baz"],
        metadatas=[
            {"split_setting": routing},
            {"split_setting": routing},
            {"split_setting": routing},
        ],
    )
    assert len(add_docs) == 3

    del_result = await routesearch.adelete(["5555", "6666", "7777", "3333", "4444"])
    logger.info(f"adelete: {del_result}")
    assert del_result["deleted"] >= 0
    # Check that default implementation of add_texts works
    results = await routesearch.aadd_texts(
        ["hello", "world"],
        ids=["3333", "44444"],
        metadatas=[{"split_setting": routing}, {"split_setting": routing}],
    )
    logger.info(f"aadd_texts: {results}")
    docs = await routesearch.aget_by_ids(["3333", "44444"])
    logger.info(f"aget_by_ids: {docs}")
    assert [doc[0].page_content for doc in docs] in [
        ["hello", "world"],
        ["world", "hello"],
    ]

    # Add texts without ids
    ids_ = await routesearch.aadd_texts(
        ["foo", "bar"],
        metadatas=[{"split_setting": routing}, {"split_setting": routing}],
    )
    assert len(ids_) == 2
    docs = await routesearch.aget_by_ids(ids_)
    assert [doc[0].page_content for doc in docs] == ["foo", "bar"]

    # Add texts with metadatas
    ids_2 = await routesearch.aadd_texts(
        ["foo", "bar"], metadatas=[{"foo": "bar", "split_setting": routing}] * 2
    )
    assert len(ids_2) == 2
    docs = await routesearch.aget_by_ids(ids_2)
    assert [doc[0].page_content for doc in docs] == ["foo", "bar"]

    # Check that add_documents works
    results = await routesearch.aadd_documents(
        [Document(id="5555", page_content="baz", metadata={"split_setting": routing})]
    )
    logger.info(f"aadd_documents: {results}")

    # Test add documents with id specified in both document and ids
    original_document = Document(
        id="7777", page_content="baz", metadata={"split_setting": routing}
    )
    results = await routesearch.aadd_documents([original_document], ids=["6666"])
    logger.info(f"aadd_documents: {results}")
    assert results == ["6666"]
    assert original_document.id == "7777"  # original document should not be modified
    results = await routesearch.aget_by_ids(["6666"])
    logger.info(f"aget_by_ids: {results}")


if __name__ == "__main__":
    route_docsearch = test_build_route_index()
    logger.info(f"route_index_name: {route_docsearch.index_name}")

    asyncio.run(async_test(route_docsearch))

    # multi query
    iterations = 1
    batch_size = 20
    multi_async_query_rrf = partial(
        test_multi_query_hybrid_rrf_search, route_docsearch, False, batch_size
    )
    elapsed_time = timeit.timeit(multi_async_query_rrf, number=iterations)
    logger.info(
        f"Async function took {elapsed_time:.10f} seconds on {iterations} queries"
    )

    multi_query_rrf = partial(
        test_multi_query_hybrid_rrf_search, route_docsearch, True, batch_size
    )
    elapsed_time = timeit.timeit(multi_query_rrf, number=iterations)
    logger.info(
        f"Sync function took {elapsed_time:.10f} seconds on {iterations} queries"
    )

    test_multi_query_hybrid_rrf_search_different_query(route_docsearch)

    # inherit codebook
    route_inherit_docsearch1 = test_inherit_codebook(True)
    logger.info(
        f"route_inherit_index_name_static: {route_inherit_docsearch1.index_name}"
    )

    route_inherit_docsearch2 = test_inherit_codebook(False)
    logger.info(
        f"route_inherit_index_name_dynamic: {route_inherit_docsearch2.index_name}"
    )

    test_route_hybrid_rrf_search(route_docsearch)
    test_route_multi_filter_with_rrf_on(route_docsearch)
    test_rout_text_search_with_must_clause(route_docsearch)

    # delete by query
    test_delete(route_docsearch)
    logger.info(f"index_name: {route_docsearch.index_name}")

    test_init()

    docsearch = test_build_index()

    # appx search
    test_appx_knn_search(docsearch)
    test_hybrid_rrf_search(docsearch)
    test_hybrid_rrf_search_withscore(docsearch)
    test_pre_filter(docsearch)
    test_post_filter(docsearch)
    test_multi_filter_with_rrf_off(docsearch)
    test_multi_filter_with_rrf_on(docsearch)
    test_search_with_min_score(docsearch)

    # text search
    test_text_search_with_must_clause(docsearch)
    test_text_search_with_must_not_clause(docsearch)
    test_text_search_with_should_clause(docsearch)
    test_text_search_with_filter_clause(docsearch)
