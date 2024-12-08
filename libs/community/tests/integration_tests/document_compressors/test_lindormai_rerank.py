"""Test Lindorm AI rerank Model."""

import asyncio
import os
from typing import Any

from langchain_core.documents import Document

from langchain_community.document_compressors.lindormai_rerank import LindormAIRerank


class Config:
    AI_LLM_ENDPOINT = os.environ.get("AI_LLM_ENDPOINT", "<LLM_ENDPOINT>")
    AI_USERNAME = os.environ.get("AI_USERNAME", "root")
    AI_PWD = os.environ.get("AI_PASSWORD", "<PASSWORD>")

    AI_DEFAULT_RERANK_MODEL = "rerank_bge_v2_m3"


reranker = LindormAIRerank(
    endpoint=Config.AI_LLM_ENDPOINT,
    username=Config.AI_USERNAME,
    password=Config.AI_PWD,
    model_name=Config.AI_DEFAULT_RERANK_MODEL,
    max_workers=5,
    client=None,
)
docs = [
    Document(page_content="量子计算是计算科学的一个前沿领域"),
    Document(page_content="预训练语言模型的发展给文本排序模型带来了新的进展"),
    Document(
        page_content="文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序"
    ),
    Document(page_content="random text for nothing"),
]


def test_rerank() -> None:
    for i, doc in enumerate(docs):
        doc.metadata = {"rating": i, "split_setting": str(i % 5)}
        doc.id = str(i)
    results = list()
    for i in range(10):
        results.append(
            reranker.compress_documents(
                query="什么是文本排序模型",
                documents=docs,
            )
        )

    assert len(results) == 10, "default top_n is 3"
    # print(f"results: {results[0]}")
    for compressed in results:
        for doc in compressed:
            assert doc.id is not None
        assert len(compressed) == 3, "default top_n is 3"
        assert compressed[0].page_content == docs[2].page_content, "rerank works"


def test_async_rerank() -> None:
    for i, doc in enumerate(docs):
        doc.metadata = {"rating": i, "split_setting": str(i % 5)}
        doc.id = str(i)

    async def async_task_demo() -> Any:
        tasks = [
            reranker.acompress_documents(query="什么是文本排序模型", documents=docs),
            reranker.acompress_documents(query="什么是文本排序模型", documents=docs),
            reranker.acompress_documents(query="什么是文本排序模型", documents=docs),
            reranker.acompress_documents(query="什么是文本排序模型", documents=docs),
            reranker.acompress_documents(query="什么是文本排序模型", documents=docs),
            reranker.acompress_documents(query="什么是文本排序模型", documents=docs),
            reranker.acompress_documents(query="什么是文本排序模型", documents=docs),
            reranker.acompress_documents(query="什么是文本排序模型", documents=docs),
            reranker.acompress_documents(query="什么是文本排序模型", documents=docs),
            reranker.acompress_documents(query="什么是文本排序模型", documents=docs),
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(async_task_demo())
    assert len(results) == 10, "default top_n is 3"
    # print(f"results: {results[0]}")
    for compressed in results:
        for doc in compressed:
            assert doc.id is not None
        assert len(compressed) == 3, "default top_n is 3"
        assert compressed[0].page_content == docs[2].page_content, "rerank works"
