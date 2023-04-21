""""""
from typing import Dict, List, Type

from pydantic import Field

from langchain.experimental.retriever_eval.base import TestRetriever
from langchain.experimental.retriever_eval.test_cases import (
    EntityLinkingTestCase,
    FirstMentionTestCase,
    LongTextOneFactTestCase,
    ManyDocsTestCase,
    RedundantDocsTestCase,
    RevisedStatementTestCase,
    SpeakerTestCase,
    TemporalQueryTestCase,
)
from langchain.experimental.retriever_eval.test_retrievers import (
    VectorStoreTestRetriever,
)
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import CharacterTextSplitter, TextSplitter


def get_test_retrievers() -> List[Type[TestRetriever]]:
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS, Chroma

    class ChromaTestRetriever(VectorStoreTestRetriever):
        base_retriever: BaseRetriever = Field(
            default_factory=lambda: Chroma(
                embedding_function=OpenAIEmbeddings()
            ).as_retriever()
        )
        text_splitter: TextSplitter = Field(
            default_factory=lambda: CharacterTextSplitter(
                chunk_size=1000, chunk_overlap=0
            )
        )
        identifying_params = {
            "chunk_size": 1000,
            "search": "similarity",
            "vectorstore": "Chroma",
            "k": 4,
        }

    class ChromaTestRetrieverMMR(VectorStoreTestRetriever):
        base_retriever: BaseRetriever = Field(
            default_factory=lambda: Chroma(
                embedding_function=OpenAIEmbeddings()
            ).as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 12})
        )
        text_splitter: TextSplitter = Field(
            default_factory=lambda: CharacterTextSplitter(
                chunk_size=200, chunk_overlap=0
            )
        )
        identifying_params = {
            "chunk_size": 200,
            "search": "mmr",
            "vectorstore": "Chroma",
            "k": 6,
        }

    class ChromaTestRetrieverStuffMetadata(ChromaTestRetrieverMMR):
        identifying_params = {
            "chunk_size": 200,
            "search": "mmr",
            "vectorstore": "Chroma",
            "k": 6,
            "metadata": "included_in_content",
        }

        def _transform_documents(self, docs: List[Document]) -> List[Document]:
            docs = super()._transform_documents(docs)
            for doc in docs:
                doc.page_content = (
                    f"Document metadata: {doc.metadata}\n\n" + doc.page_content
                )
            return docs

    class FAISSTestRetriever(VectorStoreTestRetriever):
        base_retriever: BaseRetriever = Field(
            default_factory=lambda: FAISS.from_texts(
                ["foo"], OpenAIEmbeddings()
            ).as_retriever()
        )
        text_splitter: TextSplitter = Field(
            default_factory=lambda: CharacterTextSplitter(
                chunk_size=1000, chunk_overlap=0
            )
        )
        identifying_params = {
            "chunk_size": 1000,
            "search": "similarity",
            "vectorstore": "FAISS",
            "k": 4,
        }

    return [
        ChromaTestRetriever,
        ChromaTestRetrieverMMR,
        ChromaTestRetrieverStuffMetadata,
        FAISSTestRetriever,
    ]


def get_test_cases() -> List:
    return [
        (ManyDocsTestCase, {}),
        (RedundantDocsTestCase, {}),
        (EntityLinkingTestCase, {}),
        (TemporalQueryTestCase, {}),
        (RevisedStatementTestCase, {}),
        (LongTextOneFactTestCase, {}),
        (FirstMentionTestCase, {}),
        (SpeakerTestCase, {}),
    ]


def run_test_suite() -> Dict:
    results = {}
    test_cases = get_test_cases()
    test_retrievers = get_test_retrievers()
    for retriever_cls in test_retrievers:
        retriever_name = retriever_cls().name
        results[retriever_name] = {}
        for test_case_cls, config in test_cases:
            retriever = retriever_cls()
            test_case = test_case_cls.from_config(**config)
            results[retriever_name][test_case.name] = test_case.run(retriever)
    return results
