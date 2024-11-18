"""Test Lindorm Search ByteStore."""

from langchain_community.storage.lindorm_search_bytestore import LindormSearchByteStore
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

DEFAULT_BUILD_PARAMS = {
    "lindorm_search_url": Config.SEARCH_ENDPOINT,
    "http_auth": (Config.SEARCH_USERNAME, Config.SEARCH_PWD),
    "use_ssl": False,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "bulk_size": 10000,
    "timeout": 30
}


def test_lindorm_search_bytestore(index_name="bytestore_test"):
    DEFAULT_BUILD_PARAMS["index_name"] = index_name
    bytestore = LindormSearchByteStore(**DEFAULT_BUILD_PARAMS)

    try:
        bytestore.mdelete(["k1", "k2"])
    except:
        pass

    result = bytestore.mget(["k1", "k2"])
    print(result)
    assert result == [None, None]

    bytestore.mset([("k1", b"v1"), ("k2", b"v2")])
    result = bytestore.mget(["k1", "k2"])
    print(result)
    assert result == [b"v1", b"v2"]

    bytestore.mdelete(["k1", "k2"])
    result = bytestore.mget(["k1", "k2"])
    print(result)
    assert result == [None, None]

    bytestore.client.indices.delete(index=index_name)


def test_lindorm_search_bytestore_yield_keys(index_name="bytestore_test"):
    DEFAULT_BUILD_PARAMS["index_name"] = index_name
    bytestore = LindormSearchByteStore(**DEFAULT_BUILD_PARAMS)

    bytestore.mset([("b", b"v1"),
                    ("by", b"v2"),
                    ("byte", b"v3"),
                    ("bytestore", b"v4"),
                    ("s", b"v5"),
                    ("st", b"v6"),
                    ("sto", b"v7"),
                    ("stor", b"v8"),
                    ("store", b"v9"),
                    ])
    bytestore.client.indices.refresh(index=index_name)

    prefix = "b"
    keys = list(bytestore.yield_keys(prefix=prefix))
    print(keys)
    assert len(keys) == 4

    prefix = "sto"
    keys = list(bytestore.yield_keys(prefix=prefix))
    print(keys)
    assert len(keys) == 3

    bytestore.client.indices.delete(index=index_name)


if __name__ == "__main__":
    test_lindorm_search_bytestore()
    test_lindorm_search_bytestore_yield_keys()
