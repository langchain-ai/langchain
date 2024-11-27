"""Test Lindorm AI embeddings."""

import os

from langchain_community.embeddings.lindorm_embedding import LindormAIEmbeddings


class Config:
    AI_LLM_ENDPOINT = os.environ.get("AI_ENDPOINT", "<AI_ENDPOINT>")
    AI_USERNAME = os.environ.get("AI_USERNAME", "root")
    AI_PWD = os.environ.get("AI_PASSWORD", "<PASSWORD>")

    AI_DEFAULT_EMBEDDING_MODEL = "bge_m3_model"


def test_lindormai_embedding_documents() -> None:
    documents = ["小说第一回，二女去探望郑老夫妻时，他们的酒楼生意怎样？"]
    embedding = LindormAIEmbeddings(
        endpoint=Config.AI_LLM_ENDPOINT,
        username=Config.AI_USERNAME,
        password=Config.AI_PWD,
        model_name=Config.AI_DEFAULT_EMBEDDING_MODEL,
    )  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024
    # print("embedidng:", output[0])


def test_lindormai_embedding_documents_multiple() -> None:
    documents = [
        "foo bar",
        "bar foo",
        "foo",
        "foo0",
        "foo1",
        "foo2",
        "foo3",
        "foo4",
        "foo5",
        "foo6",
        "foo7",
        "foo8",
        "foo9",
        "foo10",
        "foo11",
        "foo12",
        "foo13",
        "foo14",
        "foo15",
        "foo16",
        "foo17",
        "foo18",
        "foo19",
        "foo20",
        "foo21",
        "foo22",
        "foo23",
        "foo24",
    ]
    embedding = LindormAIEmbeddings(
        endpoint=Config.AI_LLM_ENDPOINT,
        username=Config.AI_USERNAME,
        password=Config.AI_PWD,
        model_name=Config.AI_DEFAULT_EMBEDDING_MODEL,
    )  # type: ignore[call-arg]

    output = embedding.embed_documents(documents)

    # print("embedding multi:", len(output))
    assert len(output) == 28
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024


def test_lindormai_embedding_query() -> None:
    query = "菊芬和她的姐姐是从哪个省份跑出来的？"
    embedding = LindormAIEmbeddings(
        endpoint=Config.AI_LLM_ENDPOINT,
        username=Config.AI_USERNAME,
        password=Config.AI_PWD,
        model_name=Config.AI_DEFAULT_EMBEDDING_MODEL,
    )  # type: ignore[call-arg]
    output = embedding.embed_query(query)
    assert len(output) == 1024
