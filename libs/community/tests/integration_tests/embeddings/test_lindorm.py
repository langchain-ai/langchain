"""Test Lindorm AI embeddings."""

from langchain_community.embeddings.lindormai import LindormAIEmbeddings
import environs

env = environs.Env()
env.read_env(".env")


class Config:

    AI_LLM_ENDPOINT = env.str("AI_LLM_ENDPOINT", '<LLM_ENDPOINT>')
    AI_EMB_ENDPOINT = env.str("AI_EMB_ENDPOINT", '<EMB_ENDPOINT>')
    AI_USERNAME = env.str("AI_USERNAME", 'root')
    AI_PWD = env.str("AI_PWD", '<PASSWORD>')


    AI_CHAT_LLM_ENDPOINT = env.str("AI_CHAT_LLM_ENDPOINT", '<CHAT_ENDPOINT>')
    AI_CHAT_USERNAME = env.str("AI_CHAT_USERNAME", 'root')
    AI_CHAT_PWD = env.str("AI_CHAT_PWD", '<PASSWORD>')

    AI_DEFAULT_CHAT_MODEL = "qa_model_qwen_72b_chat"
    AI_DEFAULT_RERANK_MODEL = "rerank_bge_large"
    AI_DEFAULT_EMBEDDING_MODEL = "bge-large-zh-v1.5"
    AI_DEFAULT_XIAOBU2_EMBEDDING_MODEL = "xiaobu2"
    SEARCH_ENDPOINT = env.str("SEARCH_ENDPOINT", 'SEARCH_ENDPOINT')
    SEARCH_USERNAME = env.str("SEARCH_USERNAME", 'root')
    SEARCH_PWD = env.str("SEARCH_PWD", '<PASSWORD>')


def test_lindormai_embedding_documents() -> None:
    documents = ["小说第一回，二女去探望郑老夫妻时，他们的酒楼生意怎样？"]
    embedding = LindormAIEmbeddings(endpoint=Config.AI_LLM_ENDPOINT,
                                    username=Config.AI_USERNAME,
                                    password=Config.AI_PWD,
                                    model_name=Config.AI_DEFAULT_EMBEDDING_MODEL)  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024
    print("embedidng:", output[0])


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
    embedding = LindormAIEmbeddings(endpoint=Config.AI_LLM_ENDPOINT,
                                    username=Config.AI_USERNAME,
                                    password=Config.AI_PWD,
                                    model_name=Config.AI_DEFAULT_EMBEDDING_MODEL)  # type: ignore[call-arg]

    output = embedding.embed_documents(documents)

    print("embedding multi:", len(output))
    assert len(output) == 28
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024


def test_lindormai_embedding_query() -> None:
    query = "菊芬和她的姐姐是从哪个省份跑出来的？"
    embedding = LindormAIEmbeddings(endpoint=Config.AI_LLM_ENDPOINT,
                                    username=Config.AI_USERNAME,
                                    password=Config.AI_PWD,
                                    model_name=Config.AI_DEFAULT_EMBEDDING_MODEL)  # type: ignore[call-arg]
    output = embedding.embed_query(query)

    embedding_xiaobu = LindormAIEmbeddings(endpoint=Config.AI_LLM_ENDPOINT,
                                    username=Config.AI_USERNAME,
                                    password=Config.AI_PWD,
                                    model_name=Config.AI_DEFAULT_XIAOBU2_EMBEDDING_MODEL)  # ty
    output_xiaobu = embedding_xiaobu.embed_query(query)

    print("embedding query from bge:", query, output)
    print("embedding query from xiaobu:", query, output_xiaobu)
    assert len(output) == 1024
    assert len(output_xiaobu) == 1792


if __name__ == "__main__":
    test_lindormai_embedding_documents()
    test_lindormai_embedding_documents_multiple()
    test_lindormai_embedding_query()
