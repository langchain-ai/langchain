from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel

from langchain.retrievers.document_compressors import LLMChainExtractor


def test_llm_chain_extractor() -> None:
    documents = [
        Document(
            page_content=(
                "The sky is blue. Candlepin bowling is popular in New England."
            ),
            metadata={"a": 1},
        ),
        Document(
            page_content=(
                "Mercury is the closest planet to the Sun. "
                "Candlepin bowling balls are smaller."
            ),
            metadata={"b": 2},
        ),
        Document(page_content="The moon is round.", metadata={"c": 3}),
    ]
    llm = FakeListChatModel(
        responses=[
            "Candlepin bowling is popular in New England.",
            "Candlepin bowling balls are smaller.",
            "NO_OUTPUT",
        ],
    )
    doc_compressor = LLMChainExtractor.from_llm(llm)
    output = doc_compressor.compress_documents(
        documents,
        "Tell me about Candlepin bowling.",
    )
    expected = documents = [
        Document(
            page_content="Candlepin bowling is popular in New England.",
            metadata={"a": 1},
        ),
        Document(
            page_content="Candlepin bowling balls are smaller.",
            metadata={"b": 2},
        ),
    ]
    assert output == expected


async def test_llm_chain_extractor_async() -> None:
    documents = [
        Document(
            page_content=(
                "The sky is blue. Candlepin bowling is popular in New England."
            ),
            metadata={"a": 1},
        ),
        Document(
            page_content=(
                "Mercury is the closest planet to the Sun. "
                "Candlepin bowling balls are smaller."
            ),
            metadata={"b": 2},
        ),
        Document(page_content="The moon is round.", metadata={"c": 3}),
    ]
    llm = FakeListChatModel(
        responses=[
            "Candlepin bowling is popular in New England.",
            "Candlepin bowling balls are smaller.",
            "NO_OUTPUT",
        ],
    )
    doc_compressor = LLMChainExtractor.from_llm(llm)
    output = await doc_compressor.acompress_documents(
        documents,
        "Tell me about Candlepin bowling.",
    )
    expected = [
        Document(
            page_content="Candlepin bowling is popular in New England.",
            metadata={"a": 1},
        ),
        Document(
            page_content="Candlepin bowling balls are smaller.",
            metadata={"b": 2},
        ),
    ]
    assert output == expected
