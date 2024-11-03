from langchain_core.documents import Document

from langchain_community.document_compressors.volcengine_rerank import (
    VolcengineRerank,
)


def test_rerank() -> None:
    reranker = VolcengineRerank()
    docs = [
        Document(page_content="量子计算是计算科学的一个前沿领域"),
        Document(page_content="预训练语言模型的发展给文本排序模型带来了新的进展"),
        Document(
            page_content="文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序"
        ),
        Document(page_content="random text for nothing"),
    ]
    compressed = reranker.compress_documents(
        query="什么是文本排序模型",
        documents=docs,
    )

    assert len(compressed) == 3, "default top_n is 3"
    assert compressed[0].page_content == docs[2].page_content, "rerank works"
