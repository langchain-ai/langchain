"""Test Lindorm AI rerank Model."""
import asyncio
import timeit

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_compressors.lindormai_rerank import LindormAIRerank
from langchain_community.embeddings.lindorm_embedding import LindormAIEmbeddings
from langchain_community.retrievers.lindorm_parent_document_retriever import LindormParentDocumentRetriever
from langchain_community.storage.lindorm_search_bytestore import LindormSearchByteStore
from langchain_community.vectorstores.lindorm_search_store import LindormSearchStore
from langchain.storage import _lc_store
import copy

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

reranker = LindormAIRerank(endpoint=Config.AI_LLM_ENDPOINT,
                           username=Config.AI_USERNAME,
                           password=Config.AI_PWD,
                           model_name=Config.AI_DEFAULT_RERANK_MODEL,
                           max_workers=5)
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
        results.append(reranker.compress_documents(
            query="什么是文本排序模型",
            documents=docs,
        ))

    assert len(results) == 10, "default top_n is 3"
    print(f"results: {results[0]}")
    for compressed in results:
        for doc in compressed:
            assert doc.id is not None
        assert len(compressed) == 3, "default top_n is 3"
        assert compressed[0].page_content == docs[2].page_content, "rerank works"



def test_async_rerank():
    for i, doc in enumerate(docs):
        doc.metadata = {"rating": i, "split_setting": str(i % 5)}
        doc.id = str(i)

    async def async_task_demo():
        tasks = [reranker.acompress_documents(query="什么是文本排序模型", documents=docs),
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
    print(f"results: {results[0]}")
    for compressed in results:
        for doc in compressed:
            assert doc.id is not None
        assert len(compressed) == 3, "default top_n is 3"
        assert compressed[0].page_content == docs[2].page_content, "rerank works"


# user
def parent_document(child_size, parent_size):
    from langchain_community.document_loaders import TextLoader

    loaders = [
        TextLoader("baike_documents.txt"),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    print("len:", len(docs))

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=0)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=0)

    ldai_emb = LindormAIEmbeddings(endpoint=Config.AI_LLM_ENDPOINT,
                                   username=Config.AI_USERNAME,
                                   password=Config.AI_PWD,
                                   model_name=Config.AI_DEFAULT_EMBEDDING_MODEL
                                   )

    ldv = LindormSearchStore(
        lindorm_search_url=Config.SEARCH_ENDPOINT,
        index_name="test_parent_v14",
        embedding=ldai_emb,
        http_auth=(Config.SEARCH_USERNAME, Config.SEARCH_PWD),
        timeout=60,
        method_name="ivfpq",
        routing_field="split_setting",
        nlist=12,
        analyzer="ik_max_word"
    )
    ldb = LindormSearchByteStore(
        lindorm_search_url=Config.SEARCH_ENDPOINT,
        index_name="test_parent_byte_v14",
        http_auth=(Config.SEARCH_USERNAME, Config.SEARCH_PWD),
        timeout=60,
    )
    retriever = LindormParentDocumentRetriever(vectorstore=ldv, byte_store=ldb, child_splitter=child_splitter,
                                               parent_splitter=parent_splitter)



    ROUTE = "100"
    documents = [copy.deepcopy(doc) for doc in docs for _ in range(10)]
    for i, doc in enumerate(documents):
        doc.metadata = {"rating": i, "split_setting": ROUTE, "source": "baike_document"}
        doc.id = str(i)


    kwargs = {"routing_field": "split_setting", "tag": "source", "metadata":
        {"split_setting": ROUTE}}
    retriever.add_documents(documents, **kwargs)

    retriever.search_kwargs = {"routing": ROUTE}
    retrieved_docs = retriever.invoke("辛弃疾是谁？")
    for doc in retrieved_docs:
        assert doc.id is not None, "doc id is None"
    print(f"routing: {retriever.search_kwargs}, recall: {len(retrieved_docs)}, first: {retrieved_docs[0:1]}")


    bytes = _lc_store._dump_document_as_bytes(documents[0])
    recovered = _lc_store._load_document_from_bytes(bytes)
    assert recovered == documents[0]


if __name__ == "__main__":

    iterations = 10
    elapsed_time = timeit.timeit("test_rerank()", setup="from __main__ import test_rerank", number=iterations)
    print(f"Sync function took {elapsed_time:.10f} seconds on 100 reranks")  #21s

    iterations = 10
    elapsed_time = timeit.timeit("test_async_rerank()", setup="from __main__ import test_async_rerank",
                                 number=iterations)
    print(f"Async function took {elapsed_time:.10f} seconds on 100 reranks")  #5.0529

    parent_document(400, 800)
