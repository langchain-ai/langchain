from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel

from langchain.retrievers.document_compressors import LLMChainFilter


def test_llm_chain_filter() -> None:
    documents = [
        Document(
            page_content="Candlepin bowling is popular in New England.",
            metadata={"a": 1},
        ),
        Document(
            page_content="Candlepin bowling balls are smaller.",
            metadata={"b": 2},
        ),
        Document(page_content="The moon is round.", metadata={"c": 3}),
    ]
    llm = FakeListChatModel(responses=["YES", "YES", "NO"])
    doc_compressor = LLMChainFilter.from_llm(llm)
    output = doc_compressor.compress_documents(
        documents,
        "Tell me about Candlepin bowling.",
    )
    expected = documents[:2]
    assert output == expected


async def test_llm_chain_extractor_async() -> None:
    documents = [
        Document(
            page_content="Candlepin bowling is popular in New England.",
            metadata={"a": 1},
        ),
        Document(
            page_content="Candlepin bowling balls are smaller.",
            metadata={"b": 2},
        ),
        Document(page_content="The moon is round.", metadata={"c": 3}),
    ]
    llm = FakeListChatModel(responses=["YES", "YES", "NO"])
    doc_compressor = LLMChainFilter.from_llm(llm)
    output = await doc_compressor.acompress_documents(
        documents,
        "Tell me about Candlepin bowling.",
    )
    expected = documents[:2]
    assert output == expected
