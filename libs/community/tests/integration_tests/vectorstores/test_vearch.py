# flake8: noqa

from langchain_community.vectorstores import VearchDb
from langchain_core.documents import Document
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_vearch() -> None:
    """
    Test end to end create vearch ,store vector into it and search
    """
    texts = [
        "Vearch 是一款存储大语言模型数据的向量数据库，用于存储和快速搜索模型embedding后的向量，可用于基于个人知识库的大模型应用",
        "Vearch 支持OpenAI, Llama, ChatGLM等模型，以及LangChain库",
        "vearch 是基于C语言,go语言开发的，并提供python接口，可以直接通过pip安装",
    ]
    metadatas = [
        {
            "source": (
                "/data/zhx/zhx/langchain-ChatGLM_new/knowledge_base/santi/three_body.txt"
            )
        },
        {
            "source": (
                "/data/zhx/zhx/langchain-ChatGLM_new/knowledge_base/santi/three_body.txt"
            )
        },
        {
            "source": (
                "/data/zhx/zhx/langchain-ChatGLM_new/knowledge_base/santi/three_body.txt"
            )
        },
    ]
    vearch_db = VearchDb.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        table_name="test_vearch",
        metadata_path="./",
    )
    result = vearch_db.similarity_search(
        "Vearch 支持OpenAI, Llama, ChatGLM等模型，以及LangChain库", 1
    )
    assert result == [
        Document(
            page_content="Vearch 支持OpenAI, Llama, ChatGLM等模型，以及LangChain库",
            metadata={
                "source": (
                    "/data/zhx/zhx/langchain-ChatGLM_new/knowledge_base/santi/"
                    "three_body.txt"
                )
            },
        )
    ]


def test_vearch_add_texts() -> None:
    """Test end to end adding of texts."""
    texts = [
        (
            "Vearch 是一款存储大语言模型数据的向量数据库，用于存储和快速搜索模型embedding后的向量，"
            "可用于基于个人知识库的大模型应用"
        ),
        "Vearch 支持OpenAI, Llama, ChatGLM等模型，以及LangChain库",
        "vearch 是基于C语言,go语言开发的，并提供python接口，可以直接通过pip安装",
    ]

    metadatas = [
        {
            "source": "/data/zhx/zhx/langchain-ChatGLM_new/knowledge_base/santi/"
            "three_body.txt"
        },
        {
            "source": "/data/zhx/zhx/langchain-ChatGLM_new/knowledge_base/santi/"
            "three_body.txt"
        },
        {
            "source": "/data/zhx/zhx/langchain-ChatGLM_new/knowledge_base/santi/"
            "three_body.txt"
        },
    ]
    vearch_db = VearchDb.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        table_name="test_vearch",
        metadata_path="./",
    )

    vearch_db.add_texts(
        texts=["Vearch 支持OpenAI, Llama, ChatGLM等模型，以及LangChain库"],
        metadatas=[
            {
                "source": "/data/zhx/zhx/langchain-ChatGLM_new/knowledge_base/santi/three_body.txt"
            },
        ],
    )
    result = vearch_db.similarity_search(
        "Vearch 支持OpenAI, Llama, ChatGLM等模型，以及LangChain库", 2
    )

    assert result == [
        Document(
            page_content="Vearch 支持OpenAI, Llama, ChatGLM等模型，以及LangChain库",
            metadata={
                "source": "/data/zhx/zhx/langchain-ChatGLM_new/knowledge_base/santi/three_body.txt"
            },
        ),
        Document(
            page_content="Vearch 支持OpenAI, Llama, ChatGLM等模型，以及LangChain库",
            metadata={
                "source": "/data/zhx/zhx/langchain-ChatGLM_new/knowledge_base/santi/three_body.txt"
            },
        ),
    ]
